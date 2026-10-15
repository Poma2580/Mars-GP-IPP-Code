"""Constant-memory reward distribution diagnostics."""
import math
import numpy as np


class RawTraceDiagnostics:
    """V3.1 scale diagnostics; the reservoir never changes training RNG state."""
    def __init__(self):
        self.count = 0
        self.mean = self.m2 = 0.
        self.minimum = self.maximum = None
        self.reservoir = []
        self.rng = np.random.default_rng(314159)

    def add(self, value):
        if not np.isfinite(value):
            raise FloatingPointError('Nonfinite raw Trace reward')
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = value if self.minimum is None else min(value, self.minimum)
        self.maximum = value if self.maximum is None else max(value, self.maximum)
        if len(self.reservoir) < 10000:
            self.reservoir.append(value)
        else:
            index = int(self.rng.integers(self.count))
            if index < len(self.reservoir):
                self.reservoir[index] = value

    def report(self):
        return {'count': self.count, 'mean': self.mean,
                'std': math.sqrt(max(0., self.m2 / self.count)) if self.count else None,
                'min': self.minimum, 'max': self.maximum,
                'median': float(np.median(self.reservoir)) if self.reservoir else None,
                'p95': float(np.percentile(self.reservoir, 95)) if self.reservoir else None,
                'quantile_method': 'isolated seeded reservoir, at most 10000 steps'}


class RewardDiagnostics:
    def __init__(self):
        self.groups = {group: {key: {'count': 0, 'mean': 0., 'm2': 0., 'min': None, 'max': None, 'above_one': 0}
                       for key in ('local', 'global')} for group in ('all', 'new_samples')}

    def add(self, info):
        for group, stats in self.groups.items():
            if group == 'new_samples' and not info['sampled']:
                continue
            for key, s in stats.items():
                value = info[f'r_{key}']
                s['count'] += 1
                delta = value - s['mean']
                s['mean'] += delta / s['count']
                s['m2'] += delta * (value - s['mean'])
                s['min'] = value if s['min'] is None else min(s['min'], value)
                s['max'] = value if s['max'] is None else max(s['max'], value)
                s['above_one'] += int(value > 1.)

    def report(self):
        return {group: {key: {**{k: v for k, v in s.items() if k != 'm2'},
                              'std': math.sqrt(max(0., s['m2'] / s['count'])) if s['count'] else None}
                       for key, s in stats.items()} for group, stats in self.groups.items()}
