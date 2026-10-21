"""Legal-action masks for opt-in PPO variants."""
import numpy as np
from mars_ipp.envs.actions import ACTIONS


def action_mask(position, size=16):
    candidates = np.asarray(position) + ACTIONS
    mask = ((candidates >= 0) & (candidates < size)).all(axis=1)
    if mask.shape != (8,) or not mask.any():
        raise ValueError('No legal action at this position')
    return mask


