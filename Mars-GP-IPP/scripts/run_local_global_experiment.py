"""V2 serial pipeline: reuse baselines, smoke, train, evaluate and summarize."""
import _bootstrap
import argparse
import hashlib
from pathlib import Path
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from evaluate import run as evaluate_run
from run_comparison import checkpoint_config
from mars_ipp.utils.config import ROOT, load_config, merge
from mars_ipp.utils.logger import write_json, write_csv
from mars_ipp.training.checkpoint import read_checkpoint
from mars_ipp.training.trainer import train
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.envs.observations import make_scaler
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.gp.estimator import OnlineGP
from mars_ipp.gp.uncertainty import expected_trace_reduction
from mars_ipp.evaluation.statistics import ALGORITHMS, load_runs
from mars_ipp.evaluation.local_global import METHODS, summarize_scene, validate_run
from mars_ipp.visualization.comparison import plot_results

EXPERIMENTS = ('bnorm_no_mh_local_global_3x3_rbf', 'bnorm_no_mh_local_global_3x3_rbf_no_local')
BASELINES = ROOT / 'results/comparison/bnorm_no_mh_uncertain_rbf'
OUTPUT = ROOT / 'results/comparison' / EXPERIMENTS[0]


def configs():
    return [load_config(ROOT / 'configs/main' / f'{name}.yaml') for name in EXPERIMENTS]


def preflight(root):
    cfg = configs()[0]
    fields = prepare_fields(cfg); scaler = make_scaler(fields, cfg)
    report = {'baseline_policy': 'reuse only; never reevaluate baselines', 'scenes': {}}
    with threadpool_limits(limits=1):
        for scene, field in fields.items():
            runs = load_runs([BASELINES / scene / name for name in (*ALGORITHMS, 'ours')])
            base_meta = json.loads((runs[ALGORITHMS[0]]['directory'] / 'metadata.json').read_text())
            for name, f in fields.items():
                if base_meta['fields'][name]['sha256'] != f.sha256 or base_meta['fields'][name]['parameters'] != f.parameters:
                    raise ValueError('Baseline field provenance mismatch; baseline reruns are disabled')
            env = GridEnv(field, scaler, {**cfg['env'], 'reward': cfg['reward'], 'observation_mode': cfg['ppo']['observation_mode']}, metrics_mode='terminal')
            if cfg['env'] != runs[ALGORITHMS[0]]['config']['env']:
                raise ValueError('Baseline environment differs; baseline reruns are disabled')
            gp = OnlineGP(field.parameters, 1e-6, True)
            checks = []
            for position in ([0., 0.], [10., 10.], [30., 30.], [12., 8.], [20., 16.]):
                before = float(gp.predict(env.evaluation_points)[1].sum())
                analytic = float(expected_trace_reduction(gp, np.array([position]), env.evaluation_points)[0])
                gp.add(position, field.truth(np.array([position]))[0])
                actual = before - float(gp.predict(env.evaluation_points)[1].sum())
                tolerance = max(1e-7, 1e-7 * before)
                if abs(actual - analytic) > tolerance:
                    raise ValueError('Analytic Trace mismatch')
                checks.append({'analytic': analytic, 'actual': actual, 'absolute_error': abs(actual-analytic), 'tolerance': tolerance, 'jitter': gp.jitter})
            report['scenes'][scene] = {'c_global': env.c_global, 'analytic_checks': checks,
                'baseline_sources': {k: {'directory':str(v['directory']), 'metrics_sha256':hashlib.sha256((v['directory']/'eval_metrics.csv').read_bytes()).hexdigest(), 'trajectory_sha256':hashlib.sha256((v['directory']/'trajectories.csv').read_bytes()).hexdigest()} for k,v in runs.items()}}
    write_json(root / 'preflight.json', report)
    return report


def training_dir(cfg, smoke=False):
    return ROOT / 'results/main' / (cfg['experiment'] + ('_smoke' if smoke else '')) / f"seed_{cfg['seed']}"


def ensure_training(cfg, smoke=False):
    directory = training_dir(cfg, smoke)
    if directory.exists():
        payload = read_checkpoint(directory / 'checkpoint.pt')
        if payload['config'] != cfg or not payload['progress'].get('completed') or payload['progress']['episode'] != cfg['training']['episodes']:
            raise ValueError(f'Existing incomplete/different run cannot be overwritten or resumed: {directory}')
    else:
        train(cfg, directory)
        plot_results(directory, formats=('png','pdf'))
    return directory


def smoke(root):
    reports = []
    for cfg in configs():
        cfg = merge(cfg, {'seed': 0, 'training': {'episodes': 40, 'eval_every': 10, 'checkpoint_every': 10}})
        directory = ensure_training(cfg, smoke=True)
        report = json.loads((directory / 'diagnostics.json').read_text())
        status = json.loads((directory / 'run_status.json').read_text())
        if status['pending_rollout'] != 0 or status['updates'] != 2:
            raise ValueError('Unexpected smoke rollout/update count')
        reports.append({'experiment': cfg['experiment'], 'directory': str(directory), **report})
    cfg = configs()[0]
    cfg['evaluation']['seeds'] = [1000]
    cfg['field']['test_scenario'] = 'perseverance'
    checkpoint = training_dir(cfg, smoke=True) / 'checkpoint.pt'
    benchmark_dir = root / 'smoke_formal_benchmark'
    if not benchmark_dir.exists():
        tick = time.monotonic()
        evaluate_run(cfg, benchmark_dir, checkpoint, 'ours', plot=False)
        write_json(benchmark_dir / 'timing.json', {'seconds': time.monotonic()-tick})
    formal = json.loads((benchmark_dir / 'timing.json').read_text())['seconds'] * 600
    estimates = []
    for r in reports:
        monitoring = r['monitor_seconds'] * (50/4)
        core = (r['total_seconds'] - r['monitor_seconds']) * 250
        estimates.append({'experiment':r['experiment'], 'one_run_training_seconds':core, 'one_run_monitor_seconds':monitoring, 'three_runs_seconds':3*(core+monitoring)})
    total = sum(r['three_runs_seconds'] for r in estimates) + formal
    report = {'smoke': reports, 'estimates': estimates, 'formal_evaluation_seconds': formal, 'estimated_total_seconds':total,
              'estimated_total_hours': total/3600, 'note':'40-episode smoke extrapolation; convergence/unique-sample growth can change speed; no speed ratio gate'}
    write_json(root / 'time_report.json', report)
    print(f"Estimated six training runs plus formal evaluation: {total/3600:.2f} hours", flush=True)
    if any(r['rewards']['new_samples']['global']['above_one'] for r in reports):
        raise ValueError('Global rewards above one observed; see time_report.json before formal training')
    return report


def formal_evaluation(root):
    for method, cfg in zip(METHODS, configs()):
        for seed in (0,1,2):
            own = merge(cfg, {'seed':seed})
            checkpoint = training_dir(own) / 'checkpoint.pt'
            for scene in ('zhurong','perseverance'):
                base = load_config(ROOT / 'configs/comparison/bnorm_baselines.yaml')
                base['field']['test_scenario'] = scene
                effective = checkpoint_config(checkpoint, base, 'ours')
                destination = root / scene / method / f'seed_{seed}'
                if destination.exists():
                    reference = load_runs([BASELINES / scene / a for a in ALGORITHMS])[ALGORITHMS[0]]
                    validate_run(destination, reference, seed)
                    meta = json.loads((destination/'metadata.json').read_text())
                    if meta['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
                        raise ValueError('Existing evaluation belongs to another checkpoint')
                else:
                    print(f'Evaluate {method} training seed={seed} scene={scene}', flush=True)
                    evaluate_run(effective, destination, checkpoint, 'ours', ('png','pdf'), plot=True)


def summarize(root):
    tables=[]
    for scene in ('zhurong','perseverance'):
        tables.extend(summarize_scene(root, BASELINES, scene))
    write_csv(root / 'cross_scene_summary.csv', tables)
    write_json(root / 'validation.json', {'completed':True, 'training_seeds':[0,1,2], 'formal_seeds':list(range(1000,1050)), 'formal_episodes':600,
                'baseline_policy':'all existing baseline products reused without reevaluation',
                'selection':'Perseverance RMSE and Trr equally important; conflicting metrics retain candidate status',
                'global_reward_identity':'sum over 70 actions = (trace_reset - trace_final)/C_global'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all','preflight','smoke','train','evaluate','summarize'), default='all')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--dry-run', action='store_true')
    args=parser.parse_args()
    if args.dry_run:
        print({'stage':args.stage, 'experiments':EXPERIMENTS, 'training_seeds':[0,1,2], 'monitor_seeds':list(range(1000,1005)), 'formal_seeds':list(range(1000,1050)), 'baseline_reuse':str(BASELINES), 'output':str(args.output)})
        return
    root=args.output.resolve()
    if root == BASELINES.resolve() or BASELINES.resolve() in root.parents:
        raise ValueError('V1/Baseline output directory is protected')
    root.mkdir(parents=True, exist_ok=True)
    def status(stage, completed=False, **extra):
        write_json(root / 'pipeline_status.json', {'stage':stage, 'completed':completed, **extra})
    try:
        if args.stage in ('all','preflight'):
            status('preflight'); preflight(root)
        if args.stage in ('all','smoke'):
            status('smoke'); smoke(root)
        if args.stage in ('all','train'):
            if not (root/'time_report.json').exists():
                raise ValueError('Run smoke and obtain time report before formal training')
            for cfg in configs():
                for seed in (0,1,2):
                    status('training', experiment=cfg['experiment'], seed=seed)
                    print(f"Train {cfg['experiment']} seed={seed}", flush=True)
                    ensure_training(merge(cfg, {'seed':seed}))
        if args.stage in ('all','evaluate'):
            status('formal_evaluation'); formal_evaluation(root)
        if args.stage in ('all','summarize'):
            status('summarize'); summarize(root)
        status('complete' if args.stage=='all' else args.stage+'_complete', True)
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
