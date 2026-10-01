import argparse
import copy
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv
)
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from agent import Agent
from buffer import Buffer
from trainer import Trainer

ALGO_CHOICES = ['opo']


def _serialize_arg_value(value):
    if isinstance(value, torch.device):
        return str(value)
    return value


def save_cached_scalars(run_dir, steps, values):
    np.save(os.path.join(run_dir, 'episodic_return_steps.npy'), np.asarray(steps, dtype=np.int64))
    np.save(os.path.join(run_dir, 'episodic_return_values.npy'), np.asarray(values, dtype=np.float32))


def save_hyperparameter_artifacts(run_dir, args):
    serialized_args = {key: _serialize_arg_value(value) for key, value in vars(args).items()}
    hyperparameter_text = '|param|value|\n|-|-|\n%s' % (
        '\n'.join([f'|{key}|{value}|' for key, value in serialized_args.items()])
    )
    with open(os.path.join(run_dir, 'hyperparameters.md'), 'w') as f:
        f.write(hyperparameter_text + '\n')
    with open(os.path.join(run_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(serialized_args, f, indent=2, sort_keys=True)
        f.write('\n')
    return hyperparameter_text


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=ALGO_CHOICES, default='opo')
    parser.add_argument('--envs', type=str, default=None, help='Comma-separated Atari env IDs (without NoFrameskip-v4)')
    parser.add_argument('--seeds', type=str, default=None, help='Comma-separated random seeds')
    parser.add_argument(
        '--use-cuda', '--use_cuda', dest='use_cuda',
        action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        '--torch-deterministic', '--torch_deterministic', dest='torch_deterministic',
        action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument('--total_time_steps', type=int, default=int(1e7))
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)
    parser.add_argument(
        '--learning-rate-decay', '--learning_rate_decay', dest='learning_rate_decay',
        action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--mini_batches', type=int, default=4)
    parser.add_argument('--update_epochs', type=int, default=4)
    parser.add_argument(
        '--advantage-normalization', '--advantage_normalization', dest='advantage_normalization',
        action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        '--clip-value-loss', '--clip_value_loss', dest='clip_value_loss',
        action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument('--c_1', type=float, default=0.5)
    parser.add_argument('--c_2', type=float, default=None)
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
    if args.c_2 is None:
        args.c_2 = 0.01
    if args.use_cuda:
        assert torch.cuda.is_available(), (
            '--use-cuda was requested, but CUDA is not available. '
            'Use --no-use-cuda for CPU runs, or fix the GPU/CUDA environment.'
        )
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f'Using device: {args.device}', flush=True)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.mini_batches)
    args.num_updates = int(args.total_time_steps // args.batch_size)
    return args


def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env
    return thunk


def compute_advantages(rewards, flags, values, next_value, args):
    advantages = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    adv = torch.zeros(args.num_envs).to(args.device)
    for i in reversed(range(args.num_steps)):
        returns = rewards[i] + args.gamma * flags[i] * next_value
        delta = returns - values[i]
        adv = delta + args.gamma * args.gae_lambda * flags[i] * adv
        advantages[i] = adv
        next_value = values[i]
    return advantages


def train(algo, env_id, seed, base_args=None):
    args = copy.deepcopy(base_args) if base_args is not None else get_args()
    args.env_id = env_id
    args.seed = seed
    args.algo = algo or args.algo
    min_bs = min(512, args.batch_size)
    if args.minibatch_size < min_bs:
        args.minibatch_size = min_bs
        args.mini_batches = max(1, args.batch_size // args.minibatch_size)
    if args.batch_size % args.minibatch_size != 0:
        args.minibatch_size = args.batch_size
        args.mini_batches = 1
    run_name = args.algo + '_' + str(args.epsilon) + '_resnet_seed_' + str(args.seed)
    truncation_tag = 'truncate_caps_on' if args.truncate_tail_caps else 'truncate_caps_off'
    run_name += '_' + truncation_tag
    print('[algorithm:', args.algo + ']', '[env:', args.env_id + ']', '[seed:', str(args.seed) + ']')

    # Save training logs
    path_string = str(args.env_id)[:-14] + '/' + run_name
    writer = SummaryWriter(path_string)
    run_dir = os.path.abspath(writer.log_dir)
    hyperparameter_text = save_hyperparameter_artifacts(run_dir, args)
    writer.add_text(
        'Hyperparameter',
        hyperparameter_text
    )

    # Initialize environments
    envs = gym.vector.AsyncVectorEnv([make_env(args.env_id) for _ in range(args.num_envs)])

    # State space and action space
    observation_shape = envs.single_observation_space.shape
    num_actions = envs.single_action_space.n

    # Random seed
    if args.torch_deterministic:
        numpy_rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        state, _ = envs.reset(seed=args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
    else:
        numpy_rng = np.random.default_rng()
        state, _ = envs.reset()

    # Initialize agent and optimizer
    agent = Agent(num_actions).to(args.device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)
    trainer = Trainer(args, agent, optimizer, writer)

    # Initialize buffer
    rollout_buffer = Buffer(args.num_steps, args.num_envs, observation_shape, num_actions, args.device)
    global_step = 0
    start_time = time.time()

    # This is for plotting
    episodic_returns = []
    episodic_return_steps = []
    episodic_return_values = []
    update_index = 0
    for update in tqdm(range(1, args.num_updates + 1)):

        # Linear decay of learning rate
        if args.learning_rate_decay:
            frac = 1.0 - (update - 1.0) / args.num_updates
            lr_now = frac * args.learning_rate
            optimizer.param_groups[0]['lr'] = lr_now

        for step in range(args.num_steps):
            global_step += args.num_envs

            # Compute the logarithm of the action probability output by the old policy network
            with torch.no_grad():
                action, log_prob, _, value, logits = agent.get_action_and_value(
                    torch.from_numpy(state).to(args.device).float()
                )
            action = action.cpu().numpy()

            # Update the environments
            next_state, reward, terminated, truncated, all_info = envs.step(action)

            # Save data
            flag = 1.0 - np.logical_or(terminated, truncated)
            log_prob = log_prob.cpu().numpy()
            value = value.cpu().numpy()
            logits = logits.cpu().numpy()
            rollout_buffer.push(state, action, reward, flag, log_prob, value, logits)
            state = next_state

            if 'final_info' in all_info:
                for info in all_info['final_info']:
                    if info and 'episode' in info:
                        episodic_return = float(info['episode']['r'])
                        writer.add_scalar('charts/episodic_return', episodic_return, global_step)
                        episodic_return_steps.append(global_step)
                        episodic_return_values.append(episodic_return)
                        if update // 15 == update_index:
                            episodic_returns.append(episodic_return)
                        else:
                            writer.add_scalar(
                                'This is for plotting/average_return', np.mean(episodic_returns), update_index + 1
                            )
                            episodic_returns.clear()
                            episodic_returns.append(episodic_return)
                            update_index += 1

        # ---------------------- We have collected enough data, now let's start training ---------------------- #
        states, actions, rewards, flags, log_probs, values, logits = rollout_buffer.get()

        # Use GAE technique to estimate the advantage
        with torch.no_grad():
            next_value = agent.get_value(torch.from_numpy(next_state).to(args.device).float())
            advantages = compute_advantages(rewards, flags, values, next_value, args)
            returns = advantages + values

        # Flatten each batch
        b_states = states.reshape(-1, *observation_shape)
        b_actions = actions.reshape(-1)
        b_log_probs = log_probs.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_values = values.reshape(-1)
        b_logits = logits.reshape(-1, num_actions)

        # Update the policy network and value network
        trainer.train(
            numpy_rng,
            global_step,
            b_states,
            b_actions,
            b_log_probs,
            b_advantages,
            b_returns,
            b_values,
            b_logits,
            current_update=update,
            total_updates=args.num_updates,
        )

        explained_var = (
            np.nan if torch.var(b_returns) == 0 else 1 - torch.var(b_returns - b_values) / torch.var(b_returns)
        )
        writer.add_scalar('charts/learning_rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('charts/SPS', int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar('losses/explained_variance', explained_var, global_step)

    envs.close()
    if episodic_return_values:
        save_cached_scalars(run_dir, episodic_return_steps, episodic_return_values)
    writer.close()


def _parse_csv_list(value):
    return [item.strip() for item in value.split(',') if item.strip()]


def main(algo, env_ids=None, seeds=None, base_args=None):
    if env_ids is None:
        env_ids = ['RoadRunner']
    if seeds is None:
        seeds = [1, 2, 3]
    for env_id in env_ids:
        for seed in seeds:
            train(algo, env_id + 'NoFrameskip-v4', seed, base_args=base_args)


if __name__ == '__main__':
    args = get_args()
    envs = _parse_csv_list(args.envs) if args.envs else None
    seeds = [int(s) for s in _parse_csv_list(args.seeds)] if args.seeds else None
    main(args.algo, env_ids=envs, seeds=seeds, base_args=args)
