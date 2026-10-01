import argparse
import copy
import json
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent import Agent
from trainer import Trainer

ALGO_CHOICES = ['opo']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=ALGO_CHOICES, default='opo')
    parser.add_argument('--envs', type=str, default=None, help='Comma-separated MuJoCo env IDs')
    parser.add_argument('--seeds', type=str, default=None, help='Comma-separated random seeds')
    parser.add_argument('--policy_layers', type=int, choices=[3, 7], default=7)
    parser.add_argument(
        '--use-cuda', '--use_cuda', dest='use_cuda',
        action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        '--torch-deterministic', '--torch_deterministic', dest='torch_deterministic',
        action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument('--total_time_steps', type=int, default=int(1e7))
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument(
        '--learning-rate-decay', '--learning_rate_decay', dest='learning_rate_decay',
        action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--mini_batches', type=int, choices=[1, 4, 32], default=4)
    parser.add_argument('--update_epochs', type=int, default=10)
    parser.add_argument(
        '--advantage-normalization', '--advantage_normalization', dest='advantage_normalization',
        action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        '--clip-value-loss', '--clip_value_loss', dest='clip_value_loss',
        action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument('--c_1', type=float, default=0.5)
    parser.add_argument('--c_2', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--gpd_shape', type=float, default=0.33)
    parser.add_argument('--gpd_scale', type=float, default=0.1)
    parser.add_argument(
        '--truncate-tail-caps',
        dest='truncate_tail_caps',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Truncate OPO tail caps to the empirical maximum tail value.',
    )
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.mini_batches)
    return args


def normalize_algo_name(algo):
    return algo


def save_cached_scalars(run_dir, steps, values):
    np.save(os.path.join(run_dir, 'episodic_return_steps.npy'), np.asarray(steps, dtype=np.int64))
    np.save(os.path.join(run_dir, 'episodic_return_values.npy'), np.asarray(values, dtype=np.float32))


def save_hyperparameter_artifacts(run_dir, args):
    hyperparameter_text = '|param|value|\n|-|-|\n%s' % (
        '\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()])
    )
    with open(os.path.join(run_dir, 'hyperparameters.md'), 'w') as f:
        f.write(hyperparameter_text + '\n')
    with open(os.path.join(run_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
        f.write('\n')
    return hyperparameter_text


def make_env(env_id, gamma):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda o: np.clip(o, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda r: float(np.clip(r, -10, 10)))
        return env
    return thunk


def main(env_id, seed, algo, base_args=None):
    args = copy.deepcopy(base_args) if base_args is not None else get_args()
    args.env_id = env_id
    args.seed = seed
    args.algo = normalize_algo_name(algo or args.algo)

    run_name = (
            args.algo + '_' + str(args.epsilon) +
            '_layers_' + str(args.policy_layers) +
            '_mini_bs_' + str(args.minibatch_size) +
            '_seed_' + str(args.seed)
    )
    truncation_tag = 'truncate_caps_on' if args.truncate_tail_caps else 'truncate_caps_off'
    run_name += '_' + truncation_tag
    assert args.algo in ALGO_CHOICES, 'wrong algorithm name'
    print('[algorithm:', args.algo + ']', '[env:', args.env_id + ']', '[seed:', str(args.seed) + ']')

    # Save training logs
    path_string = str(args.env_id) + '/' + run_name
    writer = SummaryWriter(path_string)
    run_dir = os.path.abspath(writer.log_dir)
    hyperparameter_text = save_hyperparameter_artifacts(run_dir, args)
    writer.add_text(
        'Hyperparameter',
        hyperparameter_text
    )

    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

    # Initialize environments
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, args.gamma) for _ in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), 'only continuous action space is supported'

    # Initialize agent
    agent = Agent(envs, args.policy_layers).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)
    trainer = Trainer(args, agent, optimizer, writer)

    # Initialize buffer
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    log_probs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    mean = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    std = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)

    # Data collection
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_time_steps // args.batch_size

    # This is for plotting
    update_index = 1
    episodic_returns = []
    episodic_return_steps = []
    episodic_return_values = []

    for update in tqdm(range(1, num_updates + 1)):

        # Linear decay of learning rate
        if args.learning_rate_decay:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * args.learning_rate
            optimizer.param_groups[0]['lr'] = lr_now

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Compute the logarithm of the action probability output by the old policy network
            with torch.no_grad():
                action, log_prob, _, value, mean_std = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            log_probs[step] = log_prob

            # Mean and standard deviation
            mean[step] = mean_std.loc
            std[step] = mean_std.scale

            # Update the environments
            next_obs, reward, terminations, truncations, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            if 'final_info' not in info:
                continue

            for item in info['final_info']:
                if item is None:
                    continue
                episodic_return = float(item['episode']['r'][0])
                writer.add_scalar('charts/episodic_return', episodic_return, global_step)
                episodic_return_steps.append(global_step)
                episodic_return_values.append(episodic_return)

                # This is for plotting
                if update == update_index:
                    episodic_returns.append(episodic_return)
                else:
                    writer.add_scalar(
                        'This is for plotting/average_return', np.mean(episodic_returns), update_index
                    )
                    episodic_returns.clear()
                    episodic_returns.append(episodic_return)
                    update_index += 1

        # Use GAE (Generalized Advantage Estimation) technique to estimate the advantage function
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            last_gae_lam = 0
            for t in reversed(range(args.num_steps)):
                next_non_terminal = 1.0 - next_done if t == args.num_steps - 1 else 1.0 - dones[t + 1]
                next_values = next_value if t == args.num_steps - 1 else values[t + 1]
                delta = rewards[t] + args.gamma * next_values * next_non_terminal - values[t]
                advantages[t] = last_gae_lam = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lam
            returns = advantages + values

        # ---------------------- We have collected enough data, now let's start training ---------------------- #
        # Flatten each batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_mean = mean.reshape(args.batch_size, -1)
        b_std = std.reshape(args.batch_size, -1)

        # Train
        trainer.train(
            global_step,
            b_obs,
            b_actions,
            b_log_probs,
            b_advantages,
            b_returns,
            b_values,
            b_mean,
            b_std,
            current_update=update,
            total_updates=num_updates,
        )

        # Save the data during the training process
        y_pre, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pre) / var_y
        writer.add_scalar('losses/explained_variance', explained_var, global_step)
        writer.add_scalar('charts/learning_rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('charts/SPS', int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    if episodic_return_values:
        save_cached_scalars(run_dir, episodic_return_steps, episodic_return_values)
    writer.close()


def _parse_csv_list(value):
    return [item.strip() for item in value.split(',') if item.strip()]


def run(algo, env_ids=None, seeds=None, base_args=None):
    if env_ids is None:
        env_ids = ['Ant-v4', 'HalfCheetah-v4', 'Hopper-v4', 'Humanoid-v4', 'HumanoidStandup-v4', 'Walker2d-v4']
    if seeds is None:
        seeds = [1, 2, 3]
    for env_id in env_ids:
        for seed in seeds:
            main(env_id, seed, algo, base_args=base_args)


if __name__ == '__main__':
    args = get_args()
    envs = _parse_csv_list(args.envs) if args.envs else None
    seeds = [int(s) for s in _parse_csv_list(args.seeds)] if args.seeds else None
    run(args.algo, env_ids=envs, seeds=seeds, base_args=args)
