"""Independent V3 variants; legacy environment and baseline behavior stay intact."""
from dataclasses import dataclass, asdict
from copy import deepcopy
import numpy as np
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.envs.observations import GPTraceScaler
from mars_ipp.evaluation.evaluator import env_config

VERSIONS = ('v3_2', 'v3_3', 'v3_4')
TRAIN_MAPS = [f'train{i}' for i in range(1, 6)]
NEW_VERSIONS = ('v3_3_4', 'v3_5')
MODES = dict(zip(VERSIONS, ('global_raw_b', 'global_relative_rep', 'global_relative_mh')))
MODES.update(v3_3_4='global_relative_rep', v3_5='global_relative_rep_mh')
REP_WEIGHT_EXPERIMENTS = {f'V3_5_{i}': ('v3_5', w) for i, w in enumerate(
    (.0275, .0412, .0550, .0687, .0825, .0962, .1099, .1237, .1374, .1512,
     .34355, .68710), 1)}
WEIGHT_EXPERIMENTS = {
    **{f'V3_3_{i}': ('v3_3', w) for i, w in enumerate((.001, .003, .005), 1)},
    **{f'V3_4_{i}': ('v3_4', w) for i, w in enumerate((.02, .05, .07, .10), 1)},
}


def validate_config(c):
    v = c['version']
    expected = 'gp_trace_b_14d' if v in ('v3_2', 'v3_5') else 'gp_trace_local_11d'
    if (c['ppo'].get('variant') != v or c['ppo']['observation_mode'] != expected
            or c['ppo']['gamma'] != .99 or c['ppo']['gae_lambda'] != .9
            or c['ppo']['update_epochs'] != 10):
        raise ValueError('Invalid variant PPO protocol')
    weights = [1.] if v == 'v3_2' else [1., 1.]
    if v in NEW_VERSIONS:
        if c['ppo'].get('action_mask') is not True:
            raise ValueError('New variants require action_mask=true')
        weights = [1., .014] if v == 'v3_3_4' else [1., .014, .02]
    experiment = c.get('experiment', '')
    if experiment == 'V3_3_3_m' and c['ppo'].get('action_mask') is not True:
        raise ValueError('V3_3_3_m requires action_mask=true')
    experiments = {**WEIGHT_EXPERIMENTS, **REP_WEIGHT_EXPERIMENTS, 'V3_3_3_m': ('v3_3', .005)}
    if experiment in experiments:
        base, weight = experiments[experiment]
        if v != base:
            raise ValueError('Weight experiment differs from base version')
        weights = [1., weight, .02] if v == 'v3_5' else [1., weight]
    elif experiment.startswith(('V3_3_', 'V3_4_', 'V3_5_')):
        raise ValueError('Unknown weight experiment')
    if c['reward'] != {'mode': MODES[v], 'weights': weights, 'disabled': []}:
        raise ValueError('Invalid variant reward protocol')
    if c['training']['maps'] != TRAIN_MAPS or set(c['field']['models']) != set(TRAIN_MAPS):
        raise ValueError('Training requires exactly train1..train5')
    if c['seed'] != 0 or c['field']['variant'] != 'original':
        raise ValueError('Require seed 0 and original fields')
    expected_env = dict(grid_size=16, coordinate_scale=2., evaluation_size=31,
                        evaluation_extent=30., max_steps=70, noise_rate=0.,
                        online_noise_variance=1e-6, unique_samples=True)
    if c['env'] != expected_env:
        raise ValueError('Invalid variant environment protocol')


@dataclass
class MagneticScaler:
    coordinate_scale: float = 15.
    mode: str = 'gp_trace_b_14d'
    d_ref: float = 0.

    def transform(self, raw):
        raw = np.asarray(raw)
        if raw.shape != (14,) or not np.isfinite(raw).all() or np.any(np.abs(raw[11:]) > 1 + 1e-7):
            raise ValueError('Expected finite normalized 14D observation')
        return np.r_[GPTraceScaler().transform(raw[:11]), raw[11:]].astype(np.float32)

    def to_dict(self):
        return asdict(self)


def scaler_for(c):
    return MagneticScaler() if c['version'] in ('v3_2', 'v3_5') else GPTraceScaler()


class VariantEnv(GridEnv):
    def __init__(self, field, scaler, config, metrics_mode='full'):
        self.variant = config['version']
        self.weight_experiment = config.get('experiment') in WEIGHT_EXPERIMENTS or config.get('experiment') == 'V3_3_3_m' or self.variant in NEW_VERSIONS
        legacy = deepcopy(config)
        legacy['ppo']['observation_mode'] = 'gp_trace_local_11d'
        legacy['reward'] = {'mode': 'gp_trace_global_raw', 'weights': [1.], 'disabled': []}
        super().__init__(field, scaler, env_config(legacy), metrics_mode)
        self.bmax = np.abs(field.truth(self.sampling_points)).max(axis=0)
        if not np.isfinite(self.bmax).all():
            raise FloatingPointError('Nonfinite per-map Bmax')
        self.bref = 2 * float(np.linalg.norm(self.bmax))
        self.reward_names = ('gp',) if self.variant == 'v3_2' else ('gp', 'rep' if self.variant == 'v3_3' else 'mh')
        if self.variant == 'v3_3_4':
            self.reward_names = ('gp', 'rep')
        elif self.variant == 'v3_5':
            self.reward_names = ('gp', 'rep', 'mh')
        self.weights = config['reward']['weights']
        self.last_raw_delta = 0.

    def _observation(self):
        base = super()._observation()
        if self.variant in ('v3_2', 'v3_5'):
            b = np.divide(self.measurement, self.bmax, out=np.zeros(3), where=self.bmax > 0)
            return np.r_[base, b]
        return base

    def _gp_trace_global_raw_step(self, next_position, outside):
        old_position, old_b = self.position.copy(), self.measurement.copy()
        repeated = not outside and tuple(next_position) in self.visited
        terms, sampled = super()._gp_trace_global_raw_step(next_position, outside)
        self.last_raw_delta = terms['gp']
        if self.variant == 'v3_2':
            return terms, sampled
        if self.variant in NEW_VERSIONS and outside:
            raise AssertionError('Masked policy selected an outside action')
        terms['gp'] /= self.trace_prior
        if self.variant in ('v3_3', 'v3_3_4', 'v3_5'):
            terms['rep'] = -float(repeated)
        if self.variant in ('v3_4', 'v3_5'):
            distance = np.linalg.norm(next_position - old_position)
            terms['mh'] = (float(np.linalg.norm(self.measurement - old_b)) / (distance * self.bref)
                           if sampled and self.bref > 0 else 0.)
        return terms, sampled

    def _info(self, stage, sampled, terms):
        # Legacy accumulation checks concern only the GP component, in its units.
        total = self.raw_reward_sum
        if self.variant != 'v3_2':
            self.raw_reward_sum *= self.trace_prior
        try:
            info = super()._info(stage, sampled, terms)
        finally:
            self.raw_reward_sum = total
        info['sum_reward'] = total
        info['raw_trace_reward'] = self.last_raw_delta if stage == 'step' else 0.
        info['raw_trace_delta'] = self.last_raw_delta if stage == 'step' else 0.
        info['b_ref'] = self.bref
        if self.weight_experiment:
            weighted = {f'weighted_r_{name}': weight * terms[name]
                        for name, weight in zip(self.reward_names, self.weights)}
            info.update(weighted)
            info['reward_total'] = sum(weighted.values())
        return info
