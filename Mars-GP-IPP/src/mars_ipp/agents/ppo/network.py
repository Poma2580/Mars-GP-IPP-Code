import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 8), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def evaluate(self, states, actions):
        distribution = torch.distributions.Categorical(self.actor(states))
        return distribution.log_prob(actions), self.critic(states).squeeze(-1), distribution.entropy()
