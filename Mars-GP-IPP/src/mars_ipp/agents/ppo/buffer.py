from dataclasses import dataclass, field
import numpy as np


def compute_gae(rewards, values, next_values, terminated, episode_ends, gamma, lam):
    """Bootstrap truncations, but never propagate advantages across episode ends."""
    rewards, values, next_values = map(lambda a: np.asarray(a, dtype=np.float64), (rewards, values, next_values))
    terminal, ends = np.asarray(terminated, dtype=bool), np.asarray(episode_ends, dtype=bool)
    advantages = np.zeros_like(rewards)
    carry = 0.
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * (not terminal[i]) * next_values[i] - values[i]
        carry = delta + gamma * lam * (not ends[i]) * carry
        advantages[i] = carry
    return advantages, values + advantages


@dataclass
class RolloutBuffer:
    transitions: list = field(default_factory=list)

    def add(self, state, action, logprob, value, reward, next_value, terminated, episode_end):
        self.transitions.append((np.asarray(state).copy(), action, logprob, value, reward, next_value, terminated, episode_end))

    def __len__(self):
        return len(self.transitions)

    def clear(self):
        self.transitions.clear()
