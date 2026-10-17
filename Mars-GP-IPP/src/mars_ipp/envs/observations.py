from dataclasses import dataclass, asdict
import numpy as np
from .actions import legal_neighbors

V3_MODE = 'gp_trace_local_11d'

LOCAL_GLOBAL_MODE = 'full_uncertainty_local_global_3x3'
SCALER_BNORM = 'signed_extrema_bnorm'
SCALER_V2 = 'signed_extrema_local_global_3x3'

def region_indices(size=16):
    if size != 16:
        raise ValueError('3x3 uncertainty requires a 16x16 grid')
    bounds = ((0, 6), (6, 11), (11, 16))
    return [np.array([x * size + y for x in range(x0, x1) for y in range(y0, y1)])
            for y0, y1 in bounds for x0, x1 in bounds]



@dataclass
class ObservationScaler:
    mean: list
    std: list
    coordinate_scale: float
    d_ref: float

    @classmethod
    def from_training_field(cls, field, size, coordinate_scale):
        positions = np.asarray([(i, j) for i in range(size) for j in range(size)])
        values = field.truth(positions * coordinate_scale)
        lookup = {tuple(p): v for p, v in zip(positions, values)}
        distances = [np.abs(lookup[tuple(q)] - value).sum()
                     for p, value in zip(positions, values)
                     for q in legal_neighbors(p, size)[1]]
        std = values.std(axis=0)
        std[std == 0] = 1.
        return cls(values.mean(axis=0).tolist(), std.tolist(), float(size - 1), float(max(distances, default=0)))

    def transform(self, raw):
        raw = np.asarray(raw, dtype=float)
        return np.concatenate((raw[:2] / self.coordinate_scale, (raw[2:] - self.mean) / self.std)).astype(np.float32)

    def to_dict(self):
        return asdict(self)


@dataclass
class SceneObservationScaler:
    scenes: dict
    coordinate_scale: float
    scene: str
    mode: str = 'signed_extrema_uncertainty'
    d_ref: float = 0.

    @classmethod
    def from_fields(cls, fields, size, coordinate_scale, scene):
        points = np.asarray([(i, j) for i in range(size) for j in range(size)]) * coordinate_scale
        scenes = {}
        for name, field in fields.items():
            values = field.truth(points)
            positive = np.maximum(values, 0).max(axis=0)
            negative = np.maximum(-values, 0).max(axis=0)
            scenes[name] = {'positive': np.where(positive > 0, positive, 1.).tolist(),
                            'negative': np.where(negative > 0, negative, 1.).tolist()}
        return cls(scenes, float(size - 1), scene)

    def for_scene(self, scene):
        if scene not in self.scenes:
            raise ValueError(f'Unknown normalization scene: {scene}')
        return SceneObservationScaler(self.scenes, self.coordinate_scale, scene, self.mode, self.d_ref)

    def transform(self, raw):
        raw = np.asarray(raw, dtype=float)
        dimension = {SCALER_V2: 22, SCALER_BNORM: 5, 'signed_extrema_uncertainty': 13}[self.mode]
        if raw.shape != (dimension,) or not np.isfinite(raw).all():
            raise ValueError(f'Expected finite {dimension}D observation')
        spec = self.scenes[self.scene]
        magnetic = raw[2:5] / np.where(raw[2:5] >= 0, spec['positive'], spec['negative'])
        return np.concatenate((raw[:2] / self.coordinate_scale, magnetic, raw[5:])).astype(np.float32)

    def to_dict(self):
        return asdict(self)


@dataclass
class GPTraceScaler:
    coordinate_scale: float = 15.
    mode: str = V3_MODE
    d_ref: float = 0.

    def transform(self, raw):
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (11,) or not np.isfinite(raw).all():
            raise ValueError('Expected finite 11D observation')
        if self.coordinate_scale != 15. or np.any(raw[:2] < 0) or np.any(raw[:2] > 15):
            raise ValueError('V3 requires grid coordinates in [0,15]')
        if np.any(raw[2:] < 0) or np.any(raw[2:] > 1):
            raise ValueError('V3 variance ratios outside [0,1]')
        return np.r_[raw[:2] / self.coordinate_scale, raw[2:]].astype(np.float32)

    def to_dict(self):
        return asdict(self)


def scaler_from_dict(spec):
    if spec.get('mode') == 'gp_trace_b_14d':
        from mars_ipp.experiments.v3_variants import MagneticScaler
        return MagneticScaler(**spec)
    if spec.get('mode') == V3_MODE:
        return GPTraceScaler(**spec)
    return SceneObservationScaler(**spec) if spec.get('mode') in ('signed_extrema_uncertainty', SCALER_BNORM, SCALER_V2) else ObservationScaler(**spec)


def make_scaler(fields, config):
    if config['ppo'].get('observation_mode') == V3_MODE:
        if config['env']['grid_size'] != 16:
            raise ValueError('V3 requires a 16x16 grid')
        return GPTraceScaler()
    if config['ppo'].get('observation_mode') in ('full_bnorm', 'full_uncertainty', LOCAL_GLOBAL_MODE):
        scaler = SceneObservationScaler.from_fields(fields, config['env']['grid_size'],
                                                  config['env']['coordinate_scale'], config['field']['train_scenario'])
        if config['ppo']['observation_mode'] == 'full_bnorm':
            scaler.mode = SCALER_BNORM
        if config['ppo']['observation_mode'] == LOCAL_GLOBAL_MODE:
            scaler.mode = SCALER_V2
        return scaler
    return ObservationScaler.from_training_field(fields[config['field']['train_scenario']],
                                                 config['env']['grid_size'], config['env']['coordinate_scale'])
