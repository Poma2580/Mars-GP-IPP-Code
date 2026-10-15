from pathlib import Path
from mars_ipp.utils.config import validate_output_path
import hashlib
import importlib.metadata
import time
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.agents.baselines.msr import Msr
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.envs.observations import make_scaler
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.evaluation.evaluator import evaluate, env_config
from mars_ipp.evaluation.statistics import summarize
from mars_ipp.utils.config import resolve_path, merge
from mars_ipp.utils.logger import append_csv, write_json
from mars_ipp.utils.seed import seed_all
from .checkpoint import save_checkpoint, model_version
from .diagnostics import RewardDiagnostics, RawTraceDiagnostics


def train(config, output=None):
    with threadpool_limits(limits=config['training']['threads']):
        return _train(config, output)


def _train(config, output):
    start = time.monotonic()
    seed_all(config['seed'])
    fields = prepare_fields(config)
    training_field = fields[config['field']['train_scenario']]
    testing_field = fields[config['field']['test_scenario']]
    scaler = make_scaler(fields, config)
    algorithm = config.get('algorithm', 'ppo')
    agent = (Msr if algorithm == 'msr' else PPO)(config['ppo'], scaler)
    env = GridEnv(training_field, scaler, env_config(config), metrics_mode='terminal')
    raw_trace = env.reward_mode == 'gp_trace_global_raw'
    directory = Path(output) if output else resolve_path(config['output_root']) / config['experiment'] / f"seed_{config['seed']}"
    directory = validate_output_path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Result directory is not empty: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    pending = {'train': [], 'eval': [], 'updates': []}
    if raw_trace:
        pending['raw_steps'] = []
    episode, total_steps = 0, 0
    env.rng = np.random.default_rng(config['seed'])
    metadata = {'fields': {name: {'source': f.source, 'sha256': f.sha256, 'parameters': f.parameters} for name, f in fields.items()},
                'scaler': scaler.to_dict(), 'versions': {n: importlib.metadata.version(n) for n in ('GPy', 'numpy', 'scipy', 'torch')},
                'device': str(agent.device), 'gpu': __import__('torch').cuda.get_device_name(agent.device) if agent.device.type == 'cuda' else None, 'eval_policy': 'categorical sampling with isolated seeded generator',
                'metric_units': {'rmse': 'nT', 'trace_current': 'nT^2', 'trr': '%'},
                'algorithm': config.get('algorithm', 'ppo'), 'm_max': env.m_max,
                'c_global': {name: GridEnv(field, scaler, env_config(config), metrics_mode='terminal').c_global for name, field in fields.items()} if env.reward_mode == 'local_global' else None,
                'region_order': 'y ascending rows, x ascending columns; [0:6,6:11,11:16]',
                'model_version': model_version(agent.input_dim),
                'source_sha256': {str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(Path(__file__).parents[1].rglob('*.py'))}}
    if env.reward_mode in ('gp_trace_relative', 'gp_trace_local_raw'):
        metadata['v3'] = config['v3']
        metadata['reward_delta'] = 'actual before/after on fixed pre-move legal neighborhood'
        metadata['reward_units'] = 'nT^2' if env.reward_mode == 'gp_trace_local_raw' else 'dimensionless'
    if raw_trace:
        metadata.update(v3=config['v3'], reward_delta='analytic on fixed 31x31 global latent variance grid',
                        reward_units='nT^2', trace_identity_tolerance={'rtol': 2e-7, 'atol': 1e-5})
    (directory / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(directory / 'metadata.json', metadata)

    def persist(completed):
        for name, filename in [('train', 'train_metrics.csv'), ('eval', 'eval_endpoints.csv'), ('updates', 'updates.csv')]:
            append_csv(directory / filename, pending[name])
        if raw_trace:
            append_csv(directory / 'raw_trace_steps.csv', pending['raw_steps'])
        save_checkpoint(directory / 'checkpoint.pt', agent, config,
                        {'episode': episode, 'steps': total_steps, 'completed': completed})
        write_json(directory / 'summary.json', summarize(pending['eval']))
        write_json(directory / 'run_status.json', {
            'episode': episode, 'steps': total_steps, 'completed': completed,
            'updates': agent.updates, 'pending_rollout': len(agent.buffer),
            'invocation_seconds': time.monotonic() - start})
        if env.reward_mode == 'local_global':
            write_json(directory / 'diagnostics.json', {**timing, 'steps': total_steps,
                       'environment_seconds_per_step': timing['environment_seconds'] / total_steps,
                       'total_seconds': time.monotonic() - start, 'rewards': reward_samples.report()})
        if raw_trace:
            write_json(directory / 'diagnostics.json', raw_diagnostics())
        for rows in pending.values():
            rows.clear()

    timing = {'environment_seconds': 0., 'monitor_seconds': 0., 'update_seconds': 0.}
    reward_samples = RewardDiagnostics()
    raw_samples = RawTraceDiagnostics() if raw_trace else None
    max_identity_error = 0.

    def raw_diagnostics():
        return {**timing, 'steps': total_steps,
                'environment_seconds_per_step': timing['environment_seconds'] / total_steps,
                'total_seconds': time.monotonic() - start, 'raw_rewards': raw_samples.report(),
                'max_abs_trace_identity_error': max_identity_error,
                'trace_identity_checked_episodes': episode,
                'parameters_finite': True, 'trace_identity_tolerance': {'rtol': 2e-7, 'atol': 1e-5}}

    def update_agent():
        report = agent.update()
        if raw_trace:
            import torch
            if (not all(np.isfinite(value) for value in report.values())
                    or not all(torch.isfinite(p).all().item() for p in agent.network.parameters())):
                raise FloatingPointError('Nonfinite V3.1 PPO update or parameters')
        return report

    while episode < config['training']['episodes']:
        observation, _ = env.reset()
        reward_sum = 0.
        episode_values = []
        terms = {name: 0. for name in env.reward_names}
        while not env.finished:
            action, logprob, value, state = agent.act(observation)
            tick = time.monotonic()
            next_observation, reward, terminated, truncated, info = env.step(action)
            timing['environment_seconds'] += time.monotonic() - tick
            if env.reward_mode == 'local_global':
                reward_samples.add(info)
            next_value = 0. if terminated else agent.value(next_observation)
            if raw_trace:
                if not np.isfinite([logprob, value, next_value, reward]).all():
                    raise FloatingPointError('Nonfinite V3.1 policy value or reward')
                episode_values.append(value)
                raw_samples.add(reward)
                pending['raw_steps'].append({'episode': episode + 1, 'step': env.count,
                    'action': action, 'i': int(env.position[0]), 'j': int(env.position[1]),
                    'sampled': info['sampled'], 'raw_trace_reward': reward,
                    'sum_reward': info['sum_reward'], 'trace_reset': info['trace_reset'],
                    'value': value, 'next_value': next_value})
            agent.buffer.add(state, action, logprob, value, reward, next_value, terminated, terminated or truncated)
            total_steps += 1
            reward_sum += reward
            for name in terms:
                terms[name] += info[f'r_{name}']
            if len(agent.buffer) >= config['ppo']['rollout_steps']:
                tick = time.monotonic()
                pending['updates'].append({'steps': total_steps, **update_agent()})
                timing['update_seconds'] += time.monotonic() - tick
            observation = next_observation
        if env.reward_mode == 'local_global':
            actual = (env.trace_reset - info['trace_current']) / env.c_global
            if not np.isclose(terms['global'], actual, rtol=2e-7, atol=1e-7):
                raise FloatingPointError('Global reward accumulation differs from reset-to-final Trace')
        episode += 1
        if raw_trace:
            max_identity_error = max(max_identity_error, abs(info['trace_identity_error']))
        pending['train'].append({'episode': episode, 'steps': total_steps, 'reward': reward_sum,
                                 **{f'r_{k}': v for k, v in terms.items()}, 'rmse': info['rmse'], 'trace_current': info['trace_current'], 'trace_prior': info['trace_prior'], 'trr': info['trr'],
                                 **({'mean_reward': reward_sum / env.count, 'unique_positions': info['unique_positions'],
                                     'repeated_visits': info['repeated_visits'], 'outside_actions': info['outside_actions']}
                                    if env.reward_mode in ('gp_trace_relative', 'gp_trace_local_raw', 'gp_trace_global_raw') else {}),
                                 **({'trace_reset': env.trace_reset, 'trace_final': info['trace_current'],
                                     'sum_reward': reward_sum, 'trace_identity_error': info['trace_identity_error'],
                                     'value_mean': float(np.mean(episode_values)),
                                     'value_min': min(episode_values), 'value_max': max(episode_values)}
                                    if raw_trace else {})})
        completed = episode == config['training']['episodes']
        if completed and len(agent.buffer):
            tick = time.monotonic()
            pending['updates'].append({'steps': total_steps, **update_agent()})
            timing['update_seconds'] += time.monotonic() - tick
        if (episode % config['training']['eval_every'] == 0
                or episode % config['training']['checkpoint_every'] == 0 or completed):
            tick = time.monotonic()
            monitor_config = merge(config, {'evaluation': {'seeds': config['training'].get('monitor_seeds', config['evaluation']['seeds'])}})
            rows, _, _ = evaluate(testing_field, scaler, monitor_config, agent,
                                  algorithm=algorithm, episode=episode, metrics_mode='terminal')
            timing['monitor_seconds'] += time.monotonic() - tick
            pending['eval'].extend(rows)
            final = summarize(rows)[0]
            print(f"episode={episode} steps={total_steps} reward={reward_sum:.4f} eval_rmse={final['rmse_mean']:.4f} trace={final['trace_current_mean']:.4f}", flush=True)
            persist(completed)
    diagnostics = {**timing, 'steps': total_steps, 'environment_seconds_per_step': timing['environment_seconds'] / total_steps, 'total_seconds': time.monotonic() - start}
    if env.reward_mode == 'local_global':
        diagnostics['rewards'] = reward_samples.report()
    if raw_trace:
        diagnostics = raw_diagnostics()
    write_json(directory / 'diagnostics.json', diagnostics)
    return directory
