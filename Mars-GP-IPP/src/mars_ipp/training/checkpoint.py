"""Atomic inference checkpoints; format 1 remains readable for evaluation."""
from pathlib import Path
import numpy as np
import torch
from mars_ipp.envs.observations import scaler_from_dict


def model_version(input_dim):
    return {11: 'ppo-11d-v3', 14: 'ppo-14d-v3', 22: 'ppo-22d-v2'}.get(input_dim, f'ppo-{input_dim}d-v1')


def save_checkpoint(path, agent, config, progress):
    _check_observation_mode(config, agent)
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save({'format': 2, 'model_version': model_version(agent.input_dim), 'input_dim': agent.input_dim,
                'algorithm': config.get('algorithm', 'ppo'),
                'network': agent.network.state_dict(), 'scaler': agent.scaler.to_dict(),
                'config': config, 'progress': progress}, temporary)
    temporary.replace(path)


def read_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') not in (1, 2):
        raise ValueError('Unsupported checkpoint: expected format 1 or 2')
    return payload


def restore_checkpoint(payload, agent):
    """Load only policy weights and the training normalization for inference."""
    _check_observation_mode(payload['config'], agent)
    if payload['config'].get('version') in ('v3_2', 'v3_3', 'v3_4'):
        from mars_ipp.experiments.v3_variants import validate_config
        validate_config(payload['config'])
        if payload.get('model_version') != model_version(agent.input_dim):
            raise ValueError('Invalid variant model version')
        if payload['config']['version'] != agent.config.get('variant'):
            raise ValueError('Checkpoint variant differs from policy')
    elif agent.input_dim == 11:
        if payload.get('model_version') != 'ppo-11d-v3':
            raise ValueError('Invalid V3 model version')
        reward = payload['config']['reward']
        if reward.get('mode') not in ('gp_trace_relative', 'gp_trace_local_raw', 'gp_trace_global_raw') or reward != {'mode': reward.get('mode'), 'weights': [1.], 'disabled': []}:
            raise ValueError('Invalid V3 checkpoint reward')
        if reward['mode'] == 'gp_trace_global_raw' and payload['config']['ppo']['gamma'] != 1.0:
            raise ValueError('Invalid V3.1 checkpoint gamma')
    if agent.input_dim == 22:
        if payload.get('model_version') != 'ppo-22d-v2':
            raise ValueError('Invalid V2 model version')
        if (payload['config']['reward'].get('mode') != 'local_global'
                or payload['config']['reward']['weights'] not in ([.25, .25, .25, .25], [0., .25, .25, .25])
                or payload['config']['reward']['weights'] != agent.config.get('reward_weights')
                or payload['config']['ppo'].get('reward_weights') != agent.config.get('reward_weights')):
            raise ValueError('Checkpoint reward variant differs from policy')
    if payload.get('input_dim', agent.input_dim) != agent.input_dim:
        raise ValueError('Checkpoint input dimension differs from policy')
    agent.load_network(payload['network'])
    if not scalers_equivalent(payload['scaler'], agent.scaler.to_dict()):
        raise ValueError('Checkpoint normalization differs from policy')
    agent.scaler = scaler_from_dict(payload['scaler'])


def _check_observation_mode(config, agent):
    mode = config['ppo'].get('observation_mode', 'full')
    if mode not in ('full', 'position_only', 'coordinates_only', 'full_bnorm', 'full_uncertainty', 'full_uncertainty_local_global_3x3', 'gp_trace_local_11d', 'gp_trace_b_14d'):
        raise ValueError(f'Invalid checkpoint observation_mode: {mode}')
    if mode != agent.observation_mode:
        raise ValueError(f'Checkpoint observation_mode {mode} differs from policy {agent.observation_mode}')


def scalers_equivalent(left, right):
    """Allow floating-point roundoff across machines, never replace stored values."""
    if set(left) != set(right) or left['coordinate_scale'] != right['coordinate_scale']:
        return False
    if left.get('mode') in ('gp_trace_local_11d', 'gp_trace_b_14d'):
        return left == right
    if 'mode' in left:
        return (left['mode'] == right['mode'] and left['scene'] == right['scene']
                and set(left['scenes']) == set(right['scenes'])
                and all(np.allclose(left['scenes'][scene][side], right['scenes'][scene][side], rtol=1e-10, atol=1e-10)
                        for scene in left['scenes'] for side in ('positive', 'negative')))
    for key in ('mean', 'std', 'd_ref'):
        a, b = np.asarray(left[key], dtype=float), np.asarray(right[key], dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            return False
        if not np.allclose(a, b, rtol=1e-10, atol=1e-10):
            return False
    return True
