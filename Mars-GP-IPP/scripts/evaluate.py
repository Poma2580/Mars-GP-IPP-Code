import _bootstrap
import argparse
import importlib.metadata
import hashlib
from pathlib import Path
import numpy as np
import yaml
from mars_ipp.training.checkpoint import read_checkpoint, restore_checkpoint, scalers_equivalent
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.agents.baselines.msr import Msr
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.envs.observations import make_scaler, scaler_from_dict
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.evaluation.evaluator import evaluate, env_config
from mars_ipp.utils.logger import write_csv, write_json
from mars_ipp.utils.config import load_config, validate_output_path, merge
from mars_ipp.utils.seed import seed_all
from mars_ipp.visualization.comparison import plot_results


def run(config, output, checkpoint=None, algorithm='ppo', formats=('png',), plot=True, preserve_checkpoint_scaler=False):
    seed_all(config['seed'])
    fields = prepare_fields(config)
    scaler = make_scaler(fields, config)
    agent = None
    episode = 0
    if checkpoint:
        payload = read_checkpoint(checkpoint)
        if preserve_checkpoint_scaler:
            scaler = scaler_from_dict(payload['scaler'])
        if not scalers_equivalent(payload['scaler'], scaler.to_dict()):
            raise ValueError('Checkpoint normalization differs from the selected field')
        agent = (Msr if algorithm == 'msr' else PPO)(config['ppo'], scaler)
        restore_checkpoint(payload, agent)
        scaler = agent.scaler
        episode = payload['progress']['episode']
    elif algorithm in ('ppo', 'msr', 'ours'):
        raise ValueError('PPO evaluation requires --checkpoint')
    output = validate_output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, trajectories, snapshot = evaluate(fields[config['field']['test_scenario']], scaler, config, agent, algorithm, episode)
    write_csv(output / 'eval_metrics.csv', rows)
    write_csv(output / 'trajectories.csv', trajectories)
    np.savez_compressed(output / 'field_snapshot.npz', **snapshot)
    (output / 'config.yaml').write_text(yaml.safe_dump(config))
    write_json(output / 'metadata.json', {'scaler': scaler.to_dict(), 'algorithm': algorithm, 'checkpoint': str(checkpoint) if checkpoint else None,
                                         'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None,
                                         'm_max': GridEnv(fields[config['field']['test_scenario']], scaler, env_config(config)).m_max,
                                         'c_global': {name: GridEnv(field, scaler, env_config(config), metrics_mode='terminal').c_global for name, field in fields.items()} if config['reward'].get('mode') == 'local_global' else None,
                                         'model_version': payload.get('model_version') if checkpoint else None,
                                         'region_order': 'y ascending rows, x ascending columns; [0:6,6:11,11:16]',
                                         'versions': {n: importlib.metadata.version(n) for n in ('GPy', 'numpy', 'scipy', 'torch')},
                                         'start_protocol': 'fixed (14,14)' if algorithm == 'cover' and not config['evaluation'].get('cover_random_start') else 'seeded random',
                                         'preserve_checkpoint_scaler': preserve_checkpoint_scaler,
                                         'checkpoint_training_config': payload['config'] if checkpoint else None,
                                         'evaluation_overrides': {key: config[key] for key in ('env', 'evaluation', 'field')},
                                         'gp_protocol': {'noise_variance': config['env'].get('online_noise_variance'),
                                                         'unique_samples': config['env'].get('unique_samples', False),
                                                         'jitter': 'GPy internal numerical factorization; recorded separately in metrics'},
                                         'source_sha256': {str(p.relative_to(_bootstrap.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                           for p in sorted((_bootstrap.ROOT/'src/mars_ipp').rglob('*.py'))},
                                         'v3': config.get('v3'), 'training_seed': config['seed'] if checkpoint else None,
                                         'reward_delta': ('analytic fixed global grid' if config['reward'].get('mode') == 'gp_trace_global_raw'
                                                          else 'actual fixed pre-move local neighborhood, nT^2' if config['reward'].get('mode') == 'gp_trace_local_raw'
                                                          else 'actual normalized' if config['reward'].get('mode') == 'gp_trace_relative' else None),
                                         'fields': {k: {'sha256': v.sha256, 'parameters': v.parameters} for k, v in fields.items()}})
    if plot:
        plot_results(output, formats=formats)
    return rows, trajectories


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(_bootstrap.ROOT / 'configs/smoke.yaml'))
    parser.add_argument('--checkpoint')
    parser.add_argument('--test-scenario', choices=('zhurong', 'perseverance'))
    parser.add_argument('--algorithm', default='ppo')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--preserve-checkpoint-scaler', action='store_true')
    parser.add_argument('--env-config', help='Override environment settings from this config; retain checkpoint policy and reward')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--formats', nargs='+', choices=('png', 'pdf'), default=['png'])
    args = parser.parse_args()
    config = read_checkpoint(args.checkpoint)['config'] if args.checkpoint else load_config(args.config)
    if args.env_config:
        config = merge(config, {'env': load_config(args.env_config)['env']})
    if args.test_scenario is not None:
        config['field']['test_scenario'] = args.test_scenario
    if args.seeds is not None:
        config = merge(config, {'evaluation': {'seeds': args.seeds}})
    run(config, args.output, args.checkpoint, args.algorithm, args.formats, plot=not args.no_plots, preserve_checkpoint_scaler=args.preserve_checkpoint_scaler)
