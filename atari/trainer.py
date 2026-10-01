import numpy as np
import torch
import torch.nn as nn
from scipy.stats import genpareto


class Trainer:
    def __init__(self, args, agent, optimizer, writer):
        self.args = args
        self.agent = agent
        self.optimizer = optimizer
        self.writer = writer

    def train(
        self,
        numpy_rng,
        global_step,
        b_obs,
        b_actions,
        b_log_probs,
        b_advantages,
        b_returns,
        b_values,
        b_old_logits=None,
        current_update=None,
        total_updates=None,
    ):
        b_index = np.arange(self.args.batch_size)

        for _epoch in range(self.args.update_epochs):
            numpy_rng.shuffle(b_index)

            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_index = b_index[start:end]

                _, new_log_prob, new_entropy, new_value, _new_logits = self.agent.get_action_and_value(
                    b_obs[mb_index], b_actions[mb_index]
                )

                log_ratio = new_log_prob - b_log_probs[mb_index]
                ratios = log_ratio.exp()

                mb_advantages = b_advantages[mb_index]
                if self.args.advantage_normalization:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                policy_loss = self._opo_loss(ratios, mb_advantages)
                value_loss = self.compute_value_loss(new_value, b_returns[mb_index], b_values[mb_index])
                entropy_loss = new_entropy.mean()
                loss = policy_loss + value_loss * self.args.c_1 - entropy_loss * self.args.c_2

                self.writer.add_scalar('charts/ratio_deviation', torch.abs(ratios - 1).mean(), global_step)
                self.writer.add_scalar('losses/policy_loss', policy_loss.item(), global_step)
                self.writer.add_scalar('losses/value_loss', value_loss.item(), global_step)
                self.writer.add_scalar('losses/entropy', entropy_loss.item(), global_step)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

    def compute_value_loss(self, new_value, mb_returns, mb_values):
        new_value = new_value.view(-1)
        if self.args.clip_value_loss:
            value_loss_unclipped = (new_value - mb_returns) ** 2
            value_clipped = mb_values + torch.clamp(new_value - mb_values, -self.args.epsilon, self.args.epsilon)
            value_loss_clipped = (value_clipped - mb_returns) ** 2
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        else:
            value_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()
        return value_loss

    def _opo_loss(self, ratios, mb_advantages):
        r = ratios
        r_safe = r.clamp_min(1e-8)
        device = r.device
        dtype = r.dtype
        truncate_tail_caps = bool(getattr(self.args, 'truncate_tail_caps', False))

        idx_plus = torch.nonzero(mb_advantages >= 0, as_tuple=False).squeeze(-1)
        idx_minus = torch.nonzero(mb_advantages < 0, as_tuple=False).squeeze(-1)

        plus_res = self._tail_caps(
            r_safe, idx_plus, group_name='positive', truncate_to_max=truncate_tail_caps
        )
        minus_res = self._tail_caps(
            1.0 / r_safe, idx_minus, group_name='negative', truncate_to_max=truncate_tail_caps
        )

        eps = torch.full_like(r, float(self.args.epsilon))

        _, plus_order, q_plus, plus_caps = plus_res
        plus_tail = plus_order[q_plus:]
        plus_idx = idx_plus[plus_tail]
        plus_caps_t = torch.tensor(plus_caps, device=device, dtype=dtype)
        eps[plus_idx] = torch.abs(plus_caps_t - 1)

        _, minus_order, q_minus, minus_caps = minus_res
        minus_tail = minus_order[q_minus:]
        minus_idx = idx_minus[minus_tail]
        minus_floor_t = torch.tensor(1.0 / minus_caps, device=device, dtype=dtype)
        eps[minus_idx] = torch.abs(1 - minus_floor_t)

        pos_mask = mb_advantages >= 0
        neg_mask = ~pos_mask
        penalty = torch.zeros_like(r)
        penalty[pos_mask] = torch.pow(r[pos_mask] - 1, 2) / (2 * eps[pos_mask])
        penalty[neg_mask] = torch.pow(r[neg_mask] - 1, 2) / (2 * eps[neg_mask])

        return -(mb_advantages * r - torch.abs(mb_advantages) * penalty).mean()

    def _tail_caps(self, values, idx, group_name, truncate_to_max=False):
        if idx.numel() < 2:
            raise ValueError(f'OPO: {group_name} group too small for tail caps (n={idx.numel()}).')

        vals = values[idx]
        vals_sorted, order = vals.sort()
        s = vals_sorted.numel()
        m = int(min(0.2 * s, 3 * np.sqrt(s)))
        if m < 1 or s - m < 1:
            raise ValueError(f'OPO: {group_name} tail size too small (n={s}, m={m}).')

        q = s - m
        threshold = np.clip(vals_sorted[q - 1].item(), 1.0, 1.0 + float(self.args.epsilon))
        threshold = np.minimum(threshold, 1.0)
        max_tail = vals_sorted[-1].item()
        truncation_max = max(max_tail, 1.0 + float(self.args.epsilon))

        shape = float(self.args.gpd_shape)
        scale = float(self.args.gpd_scale)
        p = (np.arange(1, m + 1) - 0.5) / m
        q_excess = genpareto.ppf(p, shape, loc=0, scale=scale)
        q_excess = np.nan_to_num(
            q_excess,
            nan=max_tail - threshold,
            posinf=max_tail - threshold,
            neginf=0.0,
        )
        caps = threshold + q_excess
        caps = np.maximum(caps, 1 + self.args.epsilon)
        if truncate_to_max:
            caps = np.minimum(caps, truncation_max)

        return vals_sorted, order, q, caps
