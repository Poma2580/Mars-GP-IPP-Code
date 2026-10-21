import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 8), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def distribution(self, states, action_masks=None):
        probabilities = self.actor(states)
        if action_masks is not None:
            masks = torch.as_tensor(action_masks, dtype=torch.bool, device=probabilities.device)
            if masks.shape != probabilities.shape or not masks.any(dim=-1).all():
                raise ValueError('Invalid action mask')
            probabilities = probabilities.masked_fill(~masks, 0.)
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        return torch.distributions.Categorical(probabilities)

    def evaluate(self, states, actions, action_masks=None):
        distribution = self.distribution(states, action_masks)
        return distribution.log_prob(actions), self.critic(states).squeeze(-1), distribution.entropy()
