"""YAML inheritance and paths anchored at the project root, not the shell cwd."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]


def merge(base, patch):
    result = deepcopy(base)
    for key, value in patch.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load_config(path, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"Cyclic config inheritance: {path}")
    raw = yaml.safe_load(path.read_text())
    parent = raw.pop('extends', None)
    cfg = merge(load_config(path.parent / parent, seen | {path}), raw) if parent else raw
    if cfg.get('version') in ('v3_2', 'v3_3', 'v3_4', 'v3_3_4', 'v3_5'):
        from mars_ipp.experiments.v3_variants import validate_config
        validate_config(cfg)
        return cfg
    weights = cfg['reward']['weights']
    v3 = cfg['reward'].get('mode') in ('gp_trace_relative', 'gp_trace_local_raw', 'gp_trace_global_raw')
    if v3:
        if weights != [1.] or cfg['reward']['disabled'] or cfg['ppo'].get('observation_mode') != 'gp_trace_local_11d' or not cfg['env'].get('unique_samples'):
            raise ValueError('V3 requires 11D observation, gp weight 1 and unique samples')
    if cfg['reward'].get('mode') == 'gp_trace_global_raw' and cfg['ppo']['gamma'] != 1.0:
        raise ValueError('V3.1 global raw Trace reward requires gamma=1')
    expected = 1 if v3 else 3 if cfg['reward'].get('mode') in ('msr', 'no_mh') else 4
    local_global = cfg['reward'].get('mode') == 'local_global'
    valid_v2 = weights in ([0.25, 0.25, 0.25, 0.25], [0., 0.25, 0.25, 0.25])
    if len(weights) != expected or not np.isfinite(weights).all() or min(weights) < 0 or (not valid_v2 if local_global else not np.isclose(sum(weights), 1)):
        raise ValueError('V2 requires prescribed local/global/rep/out weights' if local_global else 'Reward weights must match reward mode and sum to one')
    if not set(cfg['reward']['disabled']) <= {'mh', 'tr', 'rep', 'out'}:
        raise ValueError('Unknown disabled reward term')
    if local_global:
        if cfg['ppo'].get('observation_mode') != 'full_uncertainty_local_global_3x3' or cfg['reward']['disabled']:
            raise ValueError('V2 requires 22D state and explicit weights without disabled terms')
        cfg['ppo']['reward_weights'] = list(weights)
    return cfg


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def validate_output_path(path):
    """Keep generated artifacts beside their experiment, never in the old global folder."""
    path = Path(path)
    forbidden = (ROOT / 'results/figures').resolve()
    if path.resolve() == forbidden or forbidden in path.resolve().parents:
        raise ValueError('results/figures is retired; use the experiment output directory')
    return path
