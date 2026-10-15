"""One training/evaluation environment with explicit sampling and metric stages."""
import numpy as np
from mars_ipp.agents.base import PlanningContext
from mars_ipp.field.dataset import evaluation_grid
from mars_ipp.gp.estimator import OnlineGP
from mars_ipp.gp.uncertainty import trace, expected_trace_reduction
from .observations import LOCAL_GLOBAL_MODE, V3_MODE, region_indices

from mars_ipp.evaluation.metrics import metrics
from .actions import ACTIONS, legal_neighbors
from .rewards import reward_terms, TERMS

_GLOBAL_REFERENCES = {}


class GridEnv:
    def __init__(self, field, scaler, config, metrics_mode='full'):
        if metrics_mode not in ('full', 'terminal'):
            raise ValueError('metrics_mode must be full or terminal')
        self.metrics_mode = metrics_mode
        self.field, self.scaler = field, scaler
        self.config = config
        self.size = config['grid_size']
        self.scale = config['coordinate_scale']
        self.max_steps = config['max_steps']
        self.weights = config['reward']['weights']
        self.disabled = config['reward']['disabled']
        self.reward_mode = config['reward'].get('mode', 'full')
        if self.reward_mode not in ('full', 'msr', 'no_mh', 'local_global', 'gp_trace_relative', 'gp_trace_local_raw', 'gp_trace_global_raw'):
            raise ValueError('Unknown reward mode')
        self.reward_names = ('intensity', 'rep', 'out') if self.reward_mode == 'msr' else (('tr', 'rep', 'out') if self.reward_mode == 'no_mh' else TERMS)
        if self.reward_mode == 'local_global':
            self.reward_names = ('local', 'global', 'rep', 'out')
        self.v3_state = config.get('observation_mode') == V3_MODE
        if self.reward_mode in ('gp_trace_relative', 'gp_trace_local_raw', 'gp_trace_global_raw'):
            if not self.v3_state or self.weights != [1.] or self.disabled or not config.get('unique_samples'):
                raise ValueError('V3 requires 11D state, gp weight 1 and unique samples')
            self.reward_names = ('gp',)
        self.gp_trace_diagnostics = {}
        self.raw_reward_sum = 0.
        self.local_global_state = config.get('observation_mode') == LOCAL_GLOBAL_MODE
        self._variance_cache = None
        self.unique_samples = config.get('unique_samples', False)
        self.uncertainty_state = config.get('observation_mode') in ('full_uncertainty', LOCAL_GLOBAL_MODE)
        points = np.asarray([(i, j) for i in range(self.size) for j in range(self.size)]) * self.scale
        self.m_max = float(np.abs(field.truth(points)).sum(axis=1).max()) if self.reward_mode == 'msr' else None
        self.noise_rate = config['noise_rate']
        self.evaluation_points = evaluation_grid(config['evaluation_size'], config['evaluation_extent'])
        self.sampling_points = points
        self.regions = region_indices(self.size) if self.local_global_state else None
        self.c_global = None
        if self.reward_mode == 'local_global':
            prior_gp = OnlineGP(field.parameters, config.get('online_noise_variance'), True)
            key = (tuple(sorted(field.parameters.items())), prior_gp.noise, points.tobytes(), self.evaluation_points.tobytes())
            if key not in _GLOBAL_REFERENCES:
                value = float(expected_trace_reduction(prior_gp, points, self.evaluation_points).max())
                if not np.isfinite(value) or value <= 0:
                    raise FloatingPointError('Invalid C_global')
                _GLOBAL_REFERENCES[key] = value
            self.c_global = _GLOBAL_REFERENCES[key]
        self.reference = field.truth(self.evaluation_points)
        self.rng = np.random.default_rng()
        self.finished = True

    def _observation(self):
        if self.v3_state:
            actions, positions = legal_neighbors(self.position, self.size)
            points = np.vstack((self.position, positions)) * self.scale
            ratios = self.gp.predict(points)[1][:, 0] / self.gp.kernel.Kdiag(points)
            if not np.isfinite(ratios).all() or np.any(ratios < -1e-8) or np.any(ratios > 1 + 1e-8):
                raise FloatingPointError('Invalid V3 uncertainty features')
            neighbors = np.zeros(8)
            neighbors[actions] = np.clip(ratios[1:], 0., 1.)
            return np.r_[self.position, np.clip(ratios[0], 0., 1.), neighbors]
        raw = np.concatenate((self.position.astype(float), self.measurement))
        if not self.uncertainty_state:
            return raw
        actions, positions = legal_neighbors(self.position, self.size)
        ratios = np.zeros(8)
        points = positions * self.scale
        global_features = []
        if self.local_global_state:
            if self._variance_cache is None or self._variance_cache[0] != self.gp.revision:
                values = self.gp.predict(self.sampling_points)[1][:, 0] / self.gp.kernel.Kdiag(self.sampling_points)
                self._variance_cache = (self.gp.revision, values)
            values = self._variance_cache[1]
            ratios[actions] = values[positions[:, 0] * self.size + positions[:, 1]]
            global_features = [values[indices].mean() for indices in self.regions]
        else:
            ratios[actions] = self.gp.predict(points)[1][:, 0] / self.gp.kernel.Kdiag(points)
        if not np.isfinite(ratios).all() or np.any(ratios < -1e-8) or np.any(ratios > 1 + 1e-8):
            raise FloatingPointError('Invalid uncertainty features')
        features = np.r_[ratios, global_features]
        if not np.isfinite(features).all() or np.any(features < -1e-8) or np.any(features > 1 + 1e-8):
            raise FloatingPointError('Invalid global uncertainty features')
        return np.concatenate((raw, np.clip(features, 0., 1.)))

    def _info(self, stage, sampled, terms):
        global_metrics = {}
        if self.metrics_mode == 'full' or self.finished:
            mean, variance = self.gp.predict(self.evaluation_points)
            global_metrics = metrics(mean, variance, self.reference)
            global_metrics['trr'] = global_metrics['trace_current'] / self.trace_prior * 100
        if self.reward_mode == 'local_global':
            global_metrics['c_global'] = self.c_global
            global_metrics['trace_reset'] = getattr(self, 'trace_reset', None)
        if self.reward_mode == 'gp_trace_global_raw':
            error = None
            if self.finished:
                delta = self.trace_reset - global_metrics['trace_current']
                error = self.raw_reward_sum - delta
                if not np.isclose(self.raw_reward_sum, delta, rtol=2e-7, atol=1e-5):
                    raise FloatingPointError('Raw reward accumulation differs from reset-to-final Trace')
            global_metrics.update(trace_reset=self.trace_reset, raw_trace_reward=terms['gp'],
                                  sum_reward=self.raw_reward_sum, trace_identity_error=error)
        return {'stage': stage, 'step': self.count, 'sampled': sampled,
                'raw_observation': self._observation().copy(), 'position': self.position.copy(),
                'trace_prior': self.trace_prior, **global_metrics,
                'start_i': int(self.start_position[0]), 'start_j': int(self.start_position[1]),
                'unique_positions': len(self.visited), 'repeated_visits': self.repeated_visits,
                'outside_actions': self.outside_actions, 'coverage': len(self.visited) / self.size ** 2,
                'online_noise_variance': self.gp.noise, 'numerical_jitter': self.gp.jitter,
                **({'gp_max_score': 0., 'gp_selected_score': 0., 'gp_actual_delta': 0., 'gp_identity_error': 0., **self.gp_trace_diagnostics} if self.reward_mode in ('gp_trace_relative', 'gp_trace_local_raw') else {}),
                **{f'r_{name}': terms[name] for name in self.reward_names}, 'samples': len(self.gp.X)}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        start = (options or {}).get('start')
        self.position = self.rng.integers(0, self.size, 2) if start is None else np.asarray(start, dtype=int)
        if self.position.shape != (2,) or np.any(self.position < 0) or np.any(self.position >= self.size):
            raise ValueError('Invalid start position')
        self.start_position = self.position.copy()
        self.visited = set()
        self.repeated_visits = self.outside_actions = 0
        self.count = 0
        self.gp_trace_diagnostics = {}
        self.raw_reward_sum = 0.
        self.finished = False
        self._variance_cache = None
        self.gp = OnlineGP(self.field.parameters, self.config.get('online_noise_variance'), self.unique_samples)
        self.trace_prior = trace(self.gp.kernel.Kdiag(self.evaluation_points)[:, None])
        self.trace_reset = None
        self.measurement = np.zeros(3)
        zero = dict.fromkeys(self.reward_names, 0.)
        self.prior_info = self._info('prior', False, zero)
        self.measurement = self.field.measure(self.position * self.scale, self.rng, self.noise_rate)
        self.gp.add(self.position * self.scale, self.measurement)
        self.visited = {tuple(self.position)}
        self.trace_reset = None
        if self.reward_mode in ('local_global', 'gp_trace_global_raw'):
            self.trace_reset = trace(self.gp.predict(self.evaluation_points)[1])
        return self._observation(), self._info('reset', True, zero)

    def step(self, action):
        if self.finished:
            raise RuntimeError('Call reset before stepping a finished episode')
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 8:
            raise ValueError('Expected integer action in [0,8)')
        self.count += 1
        next_position = self.position + ACTIONS[action]
        outside = bool(np.any(next_position < 0) or np.any(next_position >= self.size))
        old = self.measurement.copy()
        sampled = False
        if self.reward_mode in ('gp_trace_relative', 'gp_trace_local_raw'):
            terms, sampled = self._gp_trace_step(action, next_position, outside)
        elif self.reward_mode == 'gp_trace_global_raw':
            terms, sampled = self._gp_trace_global_raw_step(next_position, outside)
            self.raw_reward_sum += terms['gp']
        elif self.reward_mode == 'local_global':
            self._step_sampled = False
            terms = self._local_global_step(next_position, outside)
            sampled = self._step_sampled
        elif outside:
            self.outside_actions += 1
            terms = ({'intensity': 0., 'rep': 0., 'out': -1.} if self.reward_mode == 'msr'
                     else reward_terms(old, old, self.scaler.d_ref, 0., 0., outside=True, disabled=self.disabled))
        else:
            _, neighbors = legal_neighbors(next_position, self.size, cardinal=self.reward_mode != 'no_mh')
            points = neighbors * self.scale
            before = trace(self.gp.predict(points)[1]) if self.reward_mode != 'msr' else 0.
            repeated = tuple(next_position) in self.visited
            self.position = next_position
            self.measurement = self.field.measure(self.position * self.scale, self.rng, self.noise_rate)
            self.repeated_visits += int(repeated)
            if not repeated or not self.unique_samples:
                sampled = self.gp.add(self.position * self.scale, self.measurement)
            after = trace(self.gp.predict(points)[1]) if self.reward_mode != 'msr' else 0.
            self.visited.add(tuple(self.position))
            if self.reward_mode == 'msr':
                terms = {'intensity': float(np.abs(self.measurement).sum()) / self.m_max if self.m_max > 0 else 0.,
                         'rep': -1. if repeated else 0., 'out': 0.}
            else:
                terms = reward_terms(old, self.measurement, self.scaler.d_ref, before, after, repeated, disabled=self.disabled)
        if self.reward_mode == 'no_mh':
            terms = {name: terms[name] for name in self.reward_names}
            if not sampled:
                terms['tr'] = 0.
        self.finished = self.count >= self.max_steps
        return self._observation(), float(sum(w * terms[n] for w, n in zip(self.weights, self.reward_names))), self.finished, False, self._info('step', sampled, terms)

    def _gp_trace_global_raw_step(self, next_position, outside):
        # Only the selected NEW position is scored, against the fixed global grid.
        if outside:
            self.outside_actions += 1
            return {'gp': 0.}, False
        if tuple(next_position) in self.visited:
            self.repeated_visits += 1
            self.position = next_position
            index = np.flatnonzero(np.all(self.gp.X == self.position * self.scale, axis=1))[0]
            self.measurement = self.gp.Y[index].copy()
            return {'gp': 0.}, False
        reduction = float(expected_trace_reduction(
            self.gp, (next_position * self.scale)[None, :], self.evaluation_points)[0])
        epsilon = 1e-12 * max(1., self.trace_prior)
        if not np.isfinite(reduction) or reduction < -epsilon:
            raise FloatingPointError('Invalid raw global Trace reduction')
        reduction = max(0., reduction)
        self.position = next_position
        self.measurement = self.field.measure(self.position * self.scale, self.rng, self.noise_rate)
        sampled = self.gp.add(self.position * self.scale, self.measurement)
        if not sampled:
            raise RuntimeError('New global Trace position was not sampled')
        self.visited.add(tuple(self.position))
        return {'gp': reduction}, sampled

    def _gp_trace_step(self, action, next_position, outside):
        # All scores and both variance sums use the SAME pre-move legal neighborhood.
        actions, positions = legal_neighbors(self.position, self.size)
        points = positions * self.scale
        scores = expected_trace_reduction(self.gp, points, points)
        prior_trace = float(self.gp.kernel.Kdiag(points).sum())
        epsilon = 1e-12 * max(1., prior_trace)
        if not np.isfinite(scores).all() or np.any(scores < -epsilon):
            raise FloatingPointError('Invalid GP-Trace scores')
        scores = np.maximum(scores, 0.)
        maximum = float(scores.max())
        selected = float(scores[np.flatnonzero(actions == action)[0]]) if not outside else 0.
        self.gp_trace_diagnostics = {'gp_max_score': maximum, 'gp_selected_score': selected,
                                     'gp_actual_delta': 0., 'gp_identity_error': 0.}
        if outside:
            self.outside_actions += 1
            return {'gp': 0.}, False
        repeated = tuple(next_position) in self.visited
        self.repeated_visits += int(repeated)
        if repeated:
            self.position = next_position
            # Reuse the stored measurement; revisits perform no field query or GP update.
            index = np.flatnonzero(np.all(self.gp.X == self.position * self.scale, axis=1))[0]
            self.measurement = self.gp.Y[index].copy()
            return {'gp': 0.}, False
        before = float(self.gp.predict(points)[1].sum())
        self.position = next_position
        self.measurement = self.field.measure(self.position * self.scale, self.rng, self.noise_rate)
        sampled = self.gp.add(self.position * self.scale, self.measurement)
        self.visited.add(tuple(self.position))
        actual = before - float(self.gp.predict(points)[1].sum())
        tolerance = 1e-7 * max(1., before)
        if not np.isfinite(actual) or not np.isclose(actual, selected, rtol=2e-7, atol=tolerance):
            raise FloatingPointError('GP-Trace analytic/actual reduction mismatch')
        self.gp_trace_diagnostics.update(gp_actual_delta=actual, gp_identity_error=abs(actual-selected))
        if maximum <= epsilon:
            reward = 0.
        elif self.reward_mode == 'gp_trace_local_raw':
            if actual < -tolerance:
                raise FloatingPointError('Negative raw local Trace reduction')
            reward = max(actual, 0.)
        else:
            reward = actual / maximum
            if reward < -1e-6 or reward > 1 + 1e-6:
                raise FloatingPointError('GP-Trace reward outside numerical tolerance')
        return {'gp': float(reward if self.reward_mode == 'gp_trace_local_raw' else np.clip(reward, 0., 1.))}, sampled

    def _local_global_step(self, next_position, outside):
        terms = dict.fromkeys(self.reward_names, 0.)
        if outside:
            self.outside_actions += 1
            terms['out'] = -1.
            return terms
        repeated = tuple(next_position) in self.visited
        self.repeated_visits += int(repeated)
        terms['rep'] = -1. if repeated else 0.
        will_sample = not repeated or not self.unique_samples
        if will_sample:
            _, neighbors = legal_neighbors(next_position, self.size)
            points = neighbors * self.scale
            before = trace(self.gp.predict(points)[1])
            reduction = float(expected_trace_reduction(self.gp, (next_position * self.scale)[None, :], self.evaluation_points)[0])
            if not np.isfinite(reduction) or reduction < -1e-8 * self.c_global:
                raise FloatingPointError('Invalid global Trace reduction')
        self.position = next_position
        self.measurement = self.field.measure(self.position * self.scale, self.rng, self.noise_rate)
        self._step_sampled = self.gp.add(self.position * self.scale, self.measurement) if will_sample else False
        self.visited.add(tuple(self.position))
        if self._step_sampled:
            after = trace(self.gp.predict(points)[1])
            terms['local'] = float(np.clip((before - after) / (before + 1e-12), 0., 1.))
            terms['global'] = reduction / self.c_global
        return terms

    def planning_context(self):
        actions, positions = legal_neighbors(self.position, self.size)
        return PlanningContext(self.position.copy(), actions, positions * self.scale, self.gp, self.evaluation_points)
