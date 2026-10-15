"""Evaluate four baselines; optionally add Ours using an explicit checkpoint."""
import _bootstrap
import argparse
from pathlib import Path
import json
import yaml
from evaluate import run as evaluate_run
from mars_ipp.utils.config import load_config, merge, validate_output_path
from mars_ipp.training.checkpoint import read_checkpoint
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.utils.logger import write_json
from mars_ipp.evaluation.statistics import ALGORITHMS, LABELS, load_runs, summarize_runs
from mars_ipp.visualization.comparison import plot_comparison

ROOT = _bootstrap.ROOT


def checkpoint_config(path, base, algorithm):
    """Validate protocol before creating output; keep the trained scaler and policy."""
    payload = read_checkpoint(path)
    cfg = payload['config']
    new_protocol = base['env'].get('unique_samples', False)
    ignored = {'online_noise_variance', 'unique_samples'} if new_protocol else set()
    if ({k:v for k,v in cfg['env'].items() if k not in ignored} != {k:v for k,v in base['env'].items() if k not in ignored}
            or (not new_protocol and cfg['field']['test_scenario'] != base['field']['test_scenario'])):
        raise ValueError('Checkpoint environment/test scenario differs from comparison protocol')
    if algorithm == 'msr':
        if (cfg.get('algorithm') != 'msr' or cfg['ppo'].get('observation_mode') != 'coordinates_only'
                or cfg['reward'].get('mode') != 'msr'):
            raise ValueError('Expected a genuine 2D Msr checkpoint')
        if payload['progress']['episode'] != 10000 or not payload['progress'].get('completed'):
            raise ValueError('Formal Msr evaluation requires completed 10000 Episode training')
    elif cfg.get('algorithm') == 'msr' or cfg['reward'].get('mode') == 'msr':
        raise ValueError('Ours requires its own policy and reward configuration, not Msr')
    expected = {'coordinates_only': 2, 'full_uncertainty': 13, 'full_uncertainty_local_global_3x3': 22}.get(cfg['ppo'].get('observation_mode'), 5)
    if new_protocol and algorithm == 'ours':
        if (expected, cfg['reward'].get('mode')) not in ((13, 'no_mh'), (22, 'local_global')) or cfg['env'] != base['env']:
            raise ValueError('New Ours protocol requires 13D policy, no_mh reward and matching online GP')
        if payload['progress']['episode'] != 10000 or not payload['progress'].get('completed'):
            raise ValueError('Formal Ours evaluation requires completed 10000 Episode training')
    if any(payload['network'][f'{name}.0.weight'].shape[1] != expected for name in ('actor', 'critic')):
        raise ValueError('Checkpoint network input dimension differs from observation mode')
    if expected == 22:
        if payload.get('model_version') != 'ppo-22d-v2' or cfg['ppo'].get('reward_weights') != cfg['reward']['weights']:
            raise ValueError('Invalid V2 checkpoint version/reward variant')
        if cfg['reward']['weights'] not in ([.25, .25, .25, .25], [0., .25, .25, .25]):
            raise ValueError('Invalid V2 reward weights')
    reference, actual = prepare_fields(base), prepare_fields(cfg)
    if set(reference) != set(actual) or any(reference[k].sha256 != actual[k].sha256 or reference[k].parameters != actual[k].parameters for k in reference):
        raise ValueError('Checkpoint reference field or GP parameters differ from comparison protocol')
    if new_protocol:
        return merge(cfg, {'env': base['env'], 'evaluation': base['evaluation'],
                           'field': {'test_scenario': base['field']['test_scenario']}})
    return merge(cfg, {'evaluation': {'seeds': base['evaluation']['seeds']}})


def run_comparison(config, output, msr_checkpoint, ours_checkpoint=None, stage='all', formats=('png', 'pdf')):
    output = validate_output_path(output)
    if config['env']['grid_size'] != 16 or config['env']['max_steps'] != 70 or config['field']['test_scenario'] not in ('zhurong', 'perseverance'):
        raise ValueError('Formal comparison requires the unified 16x16 scene environment and 70 actions')
    if config['comparison']['algorithms'] != list(ALGORITHMS):
        raise ValueError('Comparison must include exactly the four configured baselines')
    config = merge(config, {'evaluation': {'seeds': list(range(1000,1050))}})
    if stage in ('all', 'evaluate'):
        msr_config = checkpoint_config(msr_checkpoint, config, 'msr')
        ours_config = checkpoint_config(ours_checkpoint, config, 'ours') if ours_checkpoint else None
        algorithms = [*ALGORITHMS, *(['ours'] if ours_checkpoint else [])]
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        write_json(output / 'comparison.json', {'algorithms': algorithms, 'labels': {a: LABELS[a] for a in algorithms},
                   'msr_checkpoint': str(Path(msr_checkpoint).resolve()),
                   'ours_checkpoint': str(Path(ours_checkpoint).resolve()) if ours_checkpoint else None,
                   'completed': False})
        for algorithm in algorithms:
            print(f'Evaluating {LABELS[algorithm]}', flush=True)
            own_config = msr_config if algorithm == 'msr' else ours_config if algorithm == 'ours' else config
            checkpoint = msr_checkpoint if algorithm == 'msr' else ours_checkpoint if algorithm == 'ours' else None
            evaluate_run(own_config, output / algorithm, checkpoint, algorithm, formats, plot=False)
        manifest = json.loads((output / 'comparison.json').read_text())
        manifest['completed'] = True
        write_json(output / 'comparison.json', manifest)
    manifest_path = output / 'comparison.json'
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists() else
                {'algorithms': [*ALGORITHMS, *(['ours'] if (output/'ours').is_dir() else [])], 'completed': True})
    if not manifest['completed']:
        raise ValueError('Comparison evaluation is incomplete')
    runs = load_runs([output / a for a in manifest['algorithms']])
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    if stage in ('all', 'summarize'):
        summarize_runs(runs, output / 'summary')
    if stage in ('all', 'plot'):
        plot_comparison(runs, output / 'figures', formats)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/comparison/baselines.yaml'))
    parser.add_argument('--test-scenario', choices=('zhurong','perseverance'))
    parser.add_argument('--msr-checkpoint', default=str(ROOT/'results/main/msr/seed_0/checkpoint.pt'))
    parser.add_argument('--ours-checkpoint')
    parser.add_argument('--output', default=str(ROOT/'results/comparison/4baseline'))
    parser.add_argument('--stage', choices=('all','evaluate','summarize','plot'), default='all')
    parser.add_argument('--formats', nargs='+', choices=('png','pdf'), default=['png','pdf'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.test_scenario:
        config = merge(config, {'field': {'test_scenario': args.test_scenario}})
    if args.dry_run:
        validate_output_path(args.output)
        print({'algorithms': [*ALGORITHMS, *(['ours'] if args.ours_checkpoint else [])], 'output': args.output,
               'msr_checkpoint': args.msr_checkpoint, 'ours_checkpoint': args.ours_checkpoint, 'stage': args.stage})
    else:
        print(run_comparison(config, args.output, args.msr_checkpoint, args.ours_checkpoint, args.stage, args.formats))


if __name__ == '__main__':
    main()
