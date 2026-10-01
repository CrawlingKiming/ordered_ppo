import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from model import ResNetDeep


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.encoder = ResNetDeep()
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, state):
        return self.critic(self.encoder(state / 255.0)).squeeze(-1)

    def get_action_and_value(self, state, action=None):
        hidden = self.encoder(state / 255.0)
        distribution = Categorical(logits=self.actor(hidden))
        if action is None:
            action = distribution.sample()
        return (
            action,
            distribution.log_prob(action),
            distribution.entropy(),
            self.critic(hidden).squeeze(-1),
            distribution.logits,
        )
