import numpy as np

TERMS = ('mh', 'tr', 'rep', 'out')


def reward_terms(previous, current, d_ref, before, after, repeated=False, outside=False, disabled=()):
    if outside:
        terms = dict(zip(TERMS, (0., 0., 0., -1.)))
    else:
        distance = float(np.abs(np.asarray(current) - previous).sum())
        terms = {'mh': min(distance / d_ref, 1.) if d_ref > 0 else 0.,
                 'tr': float(np.clip((before - after) / (before + 1e-12), 0, 1)),
                 'rep': -1. if repeated else 0., 'out': 0.}
    for name in disabled:
        terms[name] = 0.
    return terms


def weighted_reward(terms, weights):
    return float(sum(weight * terms[name] for weight, name in zip(weights, TERMS)))
