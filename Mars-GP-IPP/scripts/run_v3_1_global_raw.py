"""Isolated V3.1 raw global Trace experiment. Default: seed 0 only, sequential."""
import _bootstrap
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_limits
from evaluate import run as evaluate_run
from plot_trajectories import run as plot_trajectories, load_trajectory
from run_v3_gp_trace import (sha256,
    smoke_config, ensure_training, checkpoint_for, seal_artifacts, validate_artifacts,
    validate_evaluation as validate_v3_evaluation, validate_paired_starts)
from mars_ipp.utils.config import ROOT, load_config, merge
from mars_ipp.utils.logger import write_csv, write_json
from mars_ipp.training.checkpoint import read_checkpoint
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.envs.observations import make_scaler
from mars_ipp.envs.actions import legal_neighbors
from mars_ipp.evaluation.evaluator import env_config
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.field.field_model import FieldModel
from mars_ipp.field.dataset import evaluation_grid
from mars_ipp.visualization.learning_curves import read_csv, save_figure
from mars_ipp.visualization.comparison import plot_comparison, _step_statistics

CONFIG = ROOT / 'configs/main/v3_1_gp_trace_global_raw_train1.yaml'
V3_TRAINING = ROOT / 'results/main/v3_gp_trace_train1/seed_0'
V3_COMPARISON = ROOT / 'results/comparison/v3_gp_trace_train1'
METRICS = ('rmse', 'trace_current', 'trr', 'unique_positions', 'repeated_visits', 'outside_actions')


def validate_legacy_history():
    """Honor the earlier user-requested smoke deletion without rewriting history."""
    manifest = json.loads((ROOT / 'docs/v3_protected_artifacts.json').read_text())
    missing_smoke = []
    for name, digest in manifest.items():
        path = ROOT / name
        if not path.exists() and name.startswith('results/') and any('smoke' in part.lower() for part in Path(name).parts[:-1]):
            missing_smoke.append(name)
            continue
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Protected historical artifact changed: {name}')
    return {'verified': len(manifest) - len(missing_smoke),
            'previously_deleted_smoke_files': missing_smoke,
            'exception_reason': 'Earlier explicit user request deleted all smoke results; legacy manifest retained'}


def preflight(cfg):
    baseline = yaml.safe_load((V3_TRAINING / 'config.yaml').read_text())
    checkpoint = read_checkpoint(V3_TRAINING / 'checkpoint.pt')
    if checkpoint['config'] != baseline or not checkpoint['progress']['completed']:
        raise ValueError('Formal V3 checkpoint/config mismatch or unfinished training')
    normalized = deepcopy(cfg)
    differences = {key: {'v3': old, 'v3_1': new} for key, old, new in (
        ('experiment', baseline['experiment'], cfg['experiment']),
        ('reward.mode', baseline['reward']['mode'], cfg['reward']['mode']),
        ('ppo.gamma', baseline['ppo']['gamma'], cfg['ppo']['gamma']))}
    normalized['experiment'] = baseline['experiment']
    normalized['reward']['mode'] = baseline['reward']['mode']
    normalized['ppo']['gamma'] = baseline['ppo']['gamma']
    if normalized != baseline or cfg['reward']['mode'] != 'gp_trace_global_raw' or cfg['ppo']['gamma'] != 1.:
        raise ValueError('V3.1 differs from saved V3 beyond the two allowed algorithm changes')
    fields = prepare_fields(cfg)
    if set(fields) != {'train1'}:
        raise ValueError('V3.1 must load only train1')
    field = fields['train1']
    if sha256(cfg['v3']['original_source']) != cfg['v3']['sha256'] or field.sha256 != cfg['v3']['sha256']:
        raise ValueError('train1 provenance mismatch')
    original = FieldModel.load(Path(cfg['v3']['original_source']))
    if field.parameters['type'] != 'rbf' or field.parameters != original.parameters:
        raise ValueError('train1 kernel differs from original V3')
    if cfg['ppo']['device'] == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; retain original V3 device')
        torch.ones(1, device='cuda').sum().item()
    env = GridEnv(field, make_scaler(fields, cfg), env_config(cfg), metrics_mode='terminal')
    if env.reset(seed=1000)[0].shape != (11,):
        raise ValueError('Expected unchanged 11D state')
    report = {'train1_sha256': field.sha256, 'parameters': field.parameters,
              'device': cfg['ppo']['device'], 'legacy_history': validate_legacy_history()}
    report.update(config_differences=differences, v3_training_seeds_available=[0],
                  v3_checkpoint_sha256=sha256(V3_TRAINING / 'checkpoint.pt'),
                  tolerance={'rtol': 2e-7, 'atol': 1e-5}, reward_units='nT^2')
    return report


def history_manifest(output, excluded):
    """Snapshot existing configs, data and results, including formal V3 artifacts."""
    manifest = output / 'historical_hashes.json'
    if not manifest.exists():
        files = [p for folder in ('configs', 'data/models', 'results')
                 for p in (ROOT / folder).rglob('*') if p.is_file()
                 and not any(p == e or e in p.parents for e in excluded)]
        write_json(manifest, {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)})
    verify_history(output)


def verify_history(output):
    manifest = json.loads((output / 'historical_hashes.json').read_text())
    changes = []
    failures = []
    for name, digest in manifest.items():
        path = ROOT / name
        actual = sha256(path) if path.is_file() else None
        if actual != digest:
            # Editorial reports from unrelated experiments are not comparison inputs.
            # Retain their ORIGINAL snapshot and audit the change; never bless new hashes.
            relative = Path(name)
            unrelated_report = (actual is not None and relative.name == 'report.md'
                and relative.parts[:2] == ('results', 'comparison')
                and V3_COMPARISON not in path.parents and V3_TRAINING not in path.parents)
            change = {'path': name, 'expected_sha256': digest, 'actual_sha256': actual,
                      'classification': 'unrelated_editorial_report' if unrelated_report else 'protected_artifact',
                      'modified_at_ns': path.stat().st_mtime_ns if actual is not None else None}
            changes.append(change)
            if not unrelated_report:
                failures.append(name)
    write_json(output / 'history_audit.json', {
        'snapshot_files': len(manifest), 'unchanged_files': len(manifest) - len(changes),
        'critical_checks_passed': not failures, 'changes': changes,
        'policy': 'Only existing report.md from unrelated comparison experiments is nonblocking; original hashes retained'})
    if failures:
        raise ValueError(f'Historical artifact changed: {failures[0]}')
    if changes:
        print('Historical audit: unrelated report changed; recorded in history_audit.json: '
              + ', '.join(change['path'] for change in changes), flush=True)
    return len(manifest) - len(changes)


def benchmark(cfg):
    field = prepare_fields(cfg)['train1']
    scaler = make_scaler({'train1': field}, cfg)
    cases = []
    snake = []
    for row in range(5):
        snake.extend([0 if row % 2 == 0 else 2] * 15)
        snake.append(1)
    cases.append(([0, 0], snake[:70]))
    for seed in (1000, 1001):
        rng = np.random.default_rng(seed)
        cases.append((rng.integers(0, 16, 2).tolist(), rng.integers(0, 8, 70).tolist()))
    durations = {'gp_trace_relative': [], 'gp_trace_global_raw': []}
    for repeat in range(3):
        for start, actions in cases:
            # Alternate order to reduce warm-up/order bias.
            modes = list(durations) if repeat % 2 == 0 else list(reversed(durations))
            for mode in modes:
                own = merge(cfg, {'reward': {'mode': mode}})
                env = GridEnv(field, scaler, env_config(own), metrics_mode='terminal')
                env.reset(seed=1000, options={'start': start})
                elapsed = 0.
                for action in actions:
                    tick = time.perf_counter()
                    env.step(action)
                    elapsed += time.perf_counter() - tick
                durations[mode].append(elapsed / 70)
    medians = {key: float(np.median(values)) for key, values in durations.items()}
    ratio = medians['gp_trace_global_raw'] / medians['gp_trace_relative']
    historical = json.loads((V3_TRAINING / 'diagnostics.json').read_text())['environment_seconds_per_step']
    return {'seconds_per_step_median': medians, 'paired_episode_seconds_per_step': durations,
            'ratio_raw_to_v3': ratio, 'passed': ratio <= 2.,
            'historical_v3_seconds_per_step': historical,
            'protocol': 'terminal metrics; nine matched 70-action episodes per mode; alternating order'}


def validate_raw_evaluation(directory, cfg, checkpoint):
    validate_artifacts(directory)
    if yaml.safe_load((directory / 'config.yaml').read_text()) != cfg:
        raise ValueError('V3.1 evaluation config mismatch')
    meta = json.loads((directory / 'metadata.json').read_text())
    field = prepare_fields(cfg)['train1']
    if (meta['algorithm'] != 'ppo' or meta['checkpoint_sha256'] != sha256(checkpoint)
            or meta['fields']['train1']['sha256'] != cfg['v3']['sha256']
            or meta['fields']['train1']['parameters'] != field.parameters):
        raise ValueError('V3.1 evaluation provenance mismatch')
    rows = read_csv(directory / 'eval_metrics.csv')
    if {int(r['seed']) for r in rows} != set(cfg['evaluation']['seeds']):
        raise ValueError('V3.1 evaluation seed mismatch')
    errors = []
    for seed in cfg['evaluation']['seeds']:
        selected = [r for r in rows if int(r['seed']) == seed]
        if [(r['stage'], int(r['step'])) for r in selected] != [('prior', 0), ('reset', 0)] + [('step', i) for i in range(1, 71)]:
            raise ValueError('Missing V3.1 evaluation steps')
        for row in selected:
            values = [float(row[k]) for k in (*METRICS, 'r_gp')]
            if not np.isfinite(values).all() or min(values) < 0:
                raise ValueError('Invalid V3.1 evaluation values')
            if not np.isclose(float(row['trr']), 100 * float(row['trace_current']) / float(row['trace_prior'])):
                raise ValueError('Inconsistent V3.1 Trr')
        total = sum(float(row['r_gp']) for row in selected if row['stage'] == 'step')
        delta = float(selected[1]['trace_current']) - float(selected[-1]['trace_current'])
        if not np.isclose(total, delta, rtol=2e-7, atol=1e-5):
            raise ValueError('V3.1 evaluation rewards do not telescope')
        if not np.isclose(total, float(selected[-1]['sum_reward']), rtol=2e-7, atol=1e-5):
            raise ValueError('V3.1 evaluation cumulative reward mismatch')
        errors.append(abs(total - delta))
        load_trajectory(directory, seed)
    with np.load(directory / 'field_snapshot.npz') as snapshot:
        np.testing.assert_array_equal(snapshot['points'], evaluation_grid(31, 30.))
        np.testing.assert_allclose(snapshot['reference'], field.truth(snapshot['points']), rtol=1e-10, atol=1e-10)
    return max(errors)


def smoke(cfg, training_root, output, selected):
    own = smoke_config(cfg)
    tick = time.monotonic()
    timings = []
    evaluation_errors = []
    for seed in selected:
        training = merge(own, {'seed': seed})
        checkpoint = ensure_training(training, training_root / f'seed_{seed}')
        payload = read_checkpoint(checkpoint)
        if not all(torch.isfinite(value).all().item() for value in payload['network'].values()):
            raise FloatingPointError('Nonfinite smoke checkpoint')
        test_cfg = merge(training, {'evaluation': {'seeds': [1000]}})
        directory = output / f'ppo_seed_{seed}'
        if not directory.exists():
            evaluate_run(test_cfg, directory, checkpoint, 'ppo', ('png', 'pdf'))
            seal_artifacts(directory)
        evaluation_errors.append(validate_raw_evaluation(directory, test_cfg, checkpoint))
        diag = json.loads((training_root / f'seed_{seed}' / 'diagnostics.json').read_text())
        status = json.loads((training_root / f'seed_{seed}' / 'run_status.json').read_text())
        if diag['trace_identity_checked_episodes'] != 40 or status['updates'] < 2 or not diag['parameters_finite']:
            raise ValueError('Smoke did not finish 40 checked episodes and real PPO updates')
        timings.append(diag)
    performance = benchmark(cfg)
    write_json(output / 'benchmark.json', performance)
    if not performance['passed']:
        raise RuntimeError('V3.1 environment step exceeds twice matched V3 timing; profile before training')
    report = {'completed': True, 'training_seeds': selected, 'episodes_per_training_seed': 40,
              'config_sha256': sha256(CONFIG), 'source_sha256': source_hashes(),
              'smoke_seconds': time.monotonic() - tick, 'training_diagnostics': timings,
              'max_abs_evaluation_identity_error': max(evaluation_errors), 'benchmark': performance}
    write_json(output / 'smoke_report.json', report)
    print(f'Smoke passed: seeds={selected}, raw/V3 step ratio={performance["ratio_raw_to_v3"]:.3f}', flush=True)


def source_hashes():
    files = sorted((ROOT / 'src/mars_ipp').rglob('*.py')) + [Path(__file__), ROOT / 'scripts/evaluate.py']
    return {str(path.relative_to(ROOT)): sha256(path) for path in files}


def require_smoke(output, selected):
    report = json.loads((output / 'smoke_report.json').read_text())
    if (not report['completed'] or not set(selected) <= set(report['training_seeds'])
            or report['config_sha256'] != sha256(CONFIG) or report['source_sha256'] != source_hashes()):
        raise ValueError('Run smoke for selected seeds with the current config/source before formal training')


def reuse_historical(cfg, output, method):
    source = V3_COMPARISON / ('ppo_seed_0' if method == 'v3_seed_0' else 'gp_trace')
    own = yaml.safe_load((source / 'config.yaml').read_text())
    checkpoint = V3_TRAINING / 'checkpoint.pt' if method == 'v3_seed_0' else None
    baseline = yaml.safe_load((V3_TRAINING / 'config.yaml').read_text())
    if own != baseline:
        raise ValueError('Historical evaluation config differs from formal V3')
    rows = validate_v3_evaluation(source, own, checkpoint, 'ppo' if checkpoint else 'gp_trace')
    meta = json.loads((source / 'metadata.json').read_text())
    field = prepare_fields(cfg)['train1']
    if meta['fields']['train1']['parameters'] != field.parameters:
        raise ValueError('Historical GP parameters mismatch')
    if method == 'gp_trace':
        baseline_source = 'src/mars_ipp/agents/baselines/gp_trace.py'
        if meta['source_sha256'][baseline_source] != sha256(ROOT / baseline_source):
            raise ValueError('Historical GP-Trace baseline implementation differs')
    for row in rows:
        if float(row['online_noise_variance']) != 1e-6 or int(row['step']) > 70:
            raise ValueError('Historical online GP protocol mismatch')
    with np.load(source / 'field_snapshot.npz') as snapshot:
        np.testing.assert_array_equal(snapshot['points'], evaluation_grid(31, 30.))
        np.testing.assert_allclose(snapshot['reference'], field.truth(snapshot['points']), rtol=1e-10, atol=1e-10)
    destination = output / method
    names = ('config.yaml', 'metadata.json', 'eval_metrics.csv', 'trajectories.csv', 'field_snapshot.npz')
    digests = {name: sha256(source / name) for name in names}
    if destination.exists():
        validate_artifacts(destination)
        if any(sha256(destination / name) != digest for name, digest in digests.items()):
            raise ValueError('Existing reused data differs from the source')
    else:
        destination.mkdir()
        for name in names:
            shutil.copyfile(source / name, destination / name)
        write_json(destination / 'reuse.json', {'source': source, 'source_sha256': digests,
                   'checkpoint_sha256': sha256(checkpoint) if checkpoint else None,
                   'read_only_source': True})
        # Plotting uses only the NEW copies; no writes through links into old results.
        from mars_ipp.visualization.comparison import plot_results
        plot_results(destination, formats=('png', 'pdf'))
        seal_artifacts(destination)


def ensure_evaluation(cfg, training_root, output, selected):
    for seed in selected:
        own = merge(cfg, {'seed': seed})
        checkpoint = checkpoint_for(own, training_root / f'seed_{seed}')
        directory = output / f'ppo_seed_{seed}'
        if not directory.exists():
            print(f'Evaluate V3.1 seed={seed}: 50 paired test seeds', flush=True)
            evaluate_run(own, directory, checkpoint, 'ppo', ('png', 'pdf'))
            seal_artifacts(directory)
        error = validate_raw_evaluation(directory, own, checkpoint)
        print(f'V3.1 evaluation max identity error={error:.9g}', flush=True)
    reuse_historical(cfg, output, 'v3_seed_0')
    try:
        reuse_historical(cfg, output, 'gp_trace')
    except (ValueError, KeyError, FileNotFoundError, AssertionError) as exc:
        directory = output / 'gp_trace'
        if directory.exists():
            raise ValueError('GP-Trace output already exists; refuse to overwrite') from exc
        baseline = yaml.safe_load((V3_TRAINING / 'config.yaml').read_text())
        evaluate_run(baseline, directory, None, 'gp_trace', ('png', 'pdf'))
        write_json(directory / 'retest_reason.json', {'reason': str(exc), 'baseline': 'original local 8-neighbor greedy'})
        seal_artifacts(directory)
        validate_v3_evaluation(directory, baseline, None, 'gp_trace')
    methods = ('gp_trace', 'v3_seed_0', *(f'ppo_seed_{seed}' for seed in selected))
    validate_paired_starts(output, cfg['evaluation']['seeds'], methods)
    reset_traces = []
    for method in methods:
        rows = read_csv(output / method / 'eval_metrics.csv')
        reset_traces.append({int(r['seed']): float(r['trace_current']) for r in rows if r['stage'] == 'reset'})
    for values in reset_traces[1:]:
        for seed, value in reset_traces[0].items():
            if not np.isclose(values[seed], value, rtol=1e-10, atol=1e-8):
                raise ValueError('Methods have different initial GP posteriors')
    return methods


def summarize(cfg, training_root, output, smoke_output, selected):
    methods = ('gp_trace', 'v3_seed_0', *(f'ppo_seed_{seed}' for seed in selected))
    runs = {}
    table = []
    for method in methods:
        directory = output / method
        rows = read_csv(directory / 'eval_metrics.csv')
        endpoints = [r for r in rows if r['stage'] == 'step' and int(r['step']) == 70]
        if sorted(int(r['seed']) for r in endpoints) != list(range(1000, 1050)):
            raise ValueError('Comparison requires all 50 distinct paired seeds')
        entry = {'method': method, 'training_seed': '' if method == 'gp_trace' else int(method[-1]), 'test_runs': 50}
        for key in METRICS:
            values = np.array([float(r[key]) for r in endpoints])
            entry.update({key + '_mean': float(values.mean()), key + '_std': float(values.std(ddof=0))})
        table.append(entry)
        runs[method] = {'rows': rows, 'algorithm': rows[0]['algorithm'],
                        'config': yaml.safe_load((directory / 'config.yaml').read_text())}
    summary = output / 'summary'
    write_csv(summary / 'comparison.csv', table)
    plot_comparison(runs, summary / 'figures', methods=methods)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in methods:
        steps, mean, std = _step_statistics(runs[method]['rows'], runs[method]['algorithm'], 'trace_current')
        line, = ax.plot(steps, mean, label=method)
        ax.fill_between(steps, mean - std, mean + std, color=line.get_color(), alpha=.18)
    ax.set(xlabel='Step', ylabel='Global latent Trace (nT²)')
    ax.legend(fontsize=9)
    save_figure(fig, summary / 'figures/trace_current', ('png', 'pdf'))
    for seed in (1000, 1001, 1002):
        plot_trajectories(output, seed, methods=methods)
    diagnostics = {str(seed): json.loads((training_root / f'seed_{seed}' / 'diagnostics.json').read_text()) for seed in selected}
    smoke_report = json.loads((smoke_output / 'smoke_report.json').read_text())
    validation = {'completed': True, 'training_seeds': selected, 'v3_training_seeds': [0],
                  'test_seeds': list(range(1000, 1050)), 'std_ddof': 0,
                  'checkpoint_selection': 'final completed checkpoint only',
                  'training_diagnostics': diagnostics, 'smoke': smoke_report,
                  'historical_artifacts_verified': verify_history(output)}
    validation['history_audit'] = json.loads((output / 'history_audit.json').read_text())
    write_json(summary / 'validation.json', validation)
    headers = ['方法', 'RMSE (nT)', '最终 Trace (nT²)', 'Trr (%)', '唯一点', '重复', '越界']
    report = ['# V3.1 全局原始 Trace 奖励实验', '',
        f'本轮完成 V3.1 训练 seed {selected}，每组10000回合，每组50个同地图测试种子。原 V3 仅 seed 0。',
        '累计奖励以起点采样后的 Trace_reset 为起点；gamma=1，无奖励缩放、无 Action Mask。',
        'GP-Trace 保持原合法八邻域局部贪心。标准差采用 ddof=0。',
        'train1 同地图不同起点测试不代表跨场景泛化；Trace 最优不等于 RMSE 最优。', '',
        '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in table:
        report.append('| ' + row['method'] + ' | ' + ' | '.join(
            f"{row[k + '_mean']:.4f} ± {row[k + '_std']:.4f}" for k in METRICS) + ' |')
    perf = smoke_report['benchmark']
    report += ['', f"Smoke配对环境耗时：{perf['seconds_per_step_median']}；raw/V3={perf['ratio_raw_to_v3']:.3f}。",
               '正式训练对账及奖励尺度：']
    for seed, d in diagnostics.items():
        report += [f"- seed {seed}: 10000回合；max_abs 对账误差={d['max_abs_trace_identity_error']:.9g}；"
                   f"环境耗时={1000*d['environment_seconds_per_step']:.3f} ms/step；原始奖励统计={d['raw_rewards']}。"]
    report += ['', '配置差异只有实验名、reward.mode、ppo.gamma；详见 `../preflight.json`。',
               'Smoke报告：`../../v3_1_gp_trace_global_raw_train1_smoke/smoke_report.json`（带标签运行以对应目录为准）。',
               '训练图：各 seed 目录 `figures/`；比较图：`figures/`；配对轨迹：`../figures/`。',
               '本轮未启动 seed 1/2。Critic/value loss 仅用于诊断，未据此改动 PPO。']
    if validation['history_audit']['changes']:
        report += ['', '历史审计：无关实验说明报告有内容变化，详见 `../history_audit.json`；'
                   '原始哈希快照保留，模型、配置及数据的严格校验通过。']
    (summary / 'report.md').write_text('\n'.join(report) + '\n')
    tex = [r'\begin{tabular}{lrrrrrr}', r'Method & RMSE & Trace & Trr & Unique & Repeat & Outside \\', r'\hline']
    for row in table:
        tex.append(row['method'].replace('_', r'\_') + ' & ' + ' & '.join(
            f"${row[k+'_mean']:.4f} \\pm {row[k+'_std']:.4f}$" for k in METRICS) + r' \\')
    tex.append(r'\end{tabular}')
    (summary / 'comparison.tex').write_text('\n'.join(tex) + '\n')
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('smoke', 'train', 'evaluate', 'plot', 'all'), default='all')
    parser.add_argument('--training-seeds', nargs='+', type=int, choices=(0, 1, 2), default=[0])
    parser.add_argument('--run-tag', default='')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.run_tag and not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_tag):
        parser.error('run tag must contain only letters, numbers, underscores or hyphens')
    selected = sorted(set(args.training_seeds))
    cfg = load_config(CONFIG)
    name = cfg['experiment'] + ('_' + args.run_tag if args.run_tag else '')
    training_root = ROOT / 'results/main' / name
    output = ROOT / 'results/comparison' / name
    smoke_training = ROOT / 'results/main' / (name + '_smoke')
    smoke_output = ROOT / 'results/comparison' / (name + '_smoke')
    if args.dry_run:
        print(json.dumps({'stage': args.stage, 'training_seeds': selected, 'sequential': True,
                          'training_root': str(training_root), 'evaluation_root': str(output),
                          'episodes': 10000, 'test_seeds': list(range(1000, 1050)),
                          'runtime': 'mars_gp_ipp', 'resume': False}, indent=2))
        return
    if args.stage == 'all' and any(p.exists() and any(p.iterdir()) for p in (training_root, output, smoke_training, smoke_output)):
        raise FileExistsError('Run outputs already exist; use a new run tag or explicit non-training stage')
    output.mkdir(parents=True, exist_ok=True)
    smoke_output.mkdir(parents=True, exist_ok=True)
    history_manifest(output, (training_root, output, smoke_training, smoke_output))
    started = time.monotonic()
    def status(stage, completed=False, **extra):
        write_json(output / 'pipeline_status.json', {'stage': stage, 'completed': completed,
                   'training_seeds': selected, 'invocation_seconds': time.monotonic() - started, **extra})
    with threadpool_limits(limits=1):
        try:
            status('preflight')
            write_json(output / 'preflight.json', preflight(cfg))
            if args.stage in ('all', 'smoke'):
                status('smoke')
                smoke(cfg, smoke_training, smoke_output, selected)
            if args.stage in ('all', 'train'):
                require_smoke(smoke_output, selected)
                for seed in selected:
                    status('training', training_seed=seed)
                    print(f'Train V3.1 seed={seed}, 10000 episodes', flush=True)
                    ensure_training(merge(cfg, {'seed': seed}), training_root / f'seed_{seed}')
            if args.stage in ('all', 'evaluate'):
                status('formal_evaluation')
                ensure_evaluation(cfg, training_root, output, selected)
            if args.stage in ('all', 'plot'):
                status('plot')
                summarize(cfg, training_root, output, smoke_output, selected)
            status('complete' if args.stage == 'all' else args.stage + '_complete', True,
                   historical_artifacts_verified=verify_history(output), legacy_history=validate_legacy_history())
        except Exception as exc:
            status('failed', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()
