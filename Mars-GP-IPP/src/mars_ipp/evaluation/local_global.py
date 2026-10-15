"""Multi-training-seed summaries over immutable baseline evaluations."""
from pathlib import Path
import hashlib
import json
import numpy as np
import yaml
from mars_ipp.utils.logger import write_csv, write_json
from mars_ipp.visualization.learning_curves import read_csv, save_figure
from mars_ipp.visualization.comparison import _step_statistics
from .statistics import ALGORITHMS, LABELS, load_runs

METHODS = ('v2', 'v2_global_only')
METHOD_LABELS = {'v2': 'V2', 'v2_global_only': 'V2-GlobalOnly'}
METRICS = ('rmse', 'trr', 'coverage', 'unique_positions', 'repeated_visits', 'outside_actions')


def validate_run(directory, reference, training_seed):
    directory = Path(directory)
    config = yaml.safe_load((directory / 'config.yaml').read_text())
    meta = json.loads((directory / 'metadata.json').read_text())
    ref_meta = json.loads((reference['directory'] / 'metadata.json').read_text())
    if config['seed'] != training_seed or config['env'] != reference['config']['env']:
        raise ValueError('Training seed or environment mismatch')
    if config['field']['test_scenario'] != reference['config']['field']['test_scenario'] or meta['fields'] != ref_meta['fields']:
        raise ValueError('Scene/field provenance mismatch')
    rows, paths = read_csv(directory / 'eval_metrics.csv'), read_csv(directory / 'trajectories.csv')
    if sorted({int(r['seed']) for r in rows}) != list(range(1000, 1050)):
        raise ValueError('Expected exactly 50 formal test seeds')
    starts = {int(r['seed']): (int(r['start_i']), int(r['start_j'])) for r in reference['rows'] if r['stage'] == 'reset'}
    for seed in range(1000, 1050):
        run = [r for r in rows if int(r['seed']) == seed]
        if [(r['stage'], int(r['step'])) for r in run] != [('prior', 0), ('reset', 0)] + [('step', s) for s in range(1, 71)]:
            raise ValueError('Incomplete or duplicate stages')
        if {(int(r['start_i']), int(r['start_j'])) for r in run} != {starts[seed]}:
            raise ValueError('Actual initial positions differ')
        if len({r['algorithm'] for r in run}) != 1 or any(int(r['episode']) != 10000 for r in run):
            raise ValueError('Require a single final 10000-episode checkpoint')
        for r in run:
            values = np.array([float(r[k]) for k in (*METRICS, 'trace_current', 'trace_prior')])
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('Invalid metric values')
            if not np.isclose(float(r['trr']), float(r['trace_current']) / float(r['trace_prior']) * 100):
                raise ValueError('Inconsistent Trr')
        if [int(r['step']) for r in paths if int(r['seed']) == seed] != list(range(1, 71)):
            raise ValueError('Incomplete or duplicate trajectory')
    return rows


def endpoint_stats(rows):
    final = [r for r in rows if r['stage'] == 'step' and int(r['step']) == 70]
    return {'runs': len(final), **{f'{k}_{suffix}': float(value)
            for k in METRICS for suffix, value in zip(('mean', 'std'),
            (np.mean([float(r[k]) for r in final]), np.std([float(r[k]) for r in final], ddof=0)))}}


def summarize_scene(root, baseline_root, scene, formats=('png', 'pdf')):
    root, baseline_root = Path(root), Path(baseline_root)
    baselines = load_runs([baseline_root / scene / name for name in (*ALGORITHMS, 'ours')])
    reference = baselines[ALGORITHMS[0]]
    grouped = {name: run['rows'] for name, run in baselines.items() if name != 'ours'}
    sources, per_seed, tables, combined = [], [], [], []
    for method in METHODS:
        grouped[method] = []
        identities = set()
        for training_seed in (0, 1, 2):
            directory = root / scene / method / f'seed_{training_seed}'
            rows = validate_run(directory, reference, training_seed)
            cfg = yaml.safe_load((directory / 'config.yaml').read_text())
            expected = [.25]*4 if method == 'v2' else [0., .25, .25, .25]
            if cfg['reward']['mode'] != 'local_global' or cfg['reward']['weights'] != expected or cfg['ppo']['update_epochs'] != 10:
                raise ValueError('Wrong V2 variant or PPO update epochs')
            per_seed.append({'scenario': scene, 'method': METHOD_LABELS[method], 'training_seed': training_seed, **endpoint_stats(rows)})
            for row in rows:
                identity = (method, training_seed, int(row['seed']), scene, row['stage'], int(row['step']))
                if identity in identities:
                    raise ValueError('Duplicate run identity')
                identities.add(identity)
                row = {**row, 'training_seed': training_seed, 'scenario': scene, 'algorithm': method}
                grouped[method].append(row); combined.append(row)
            sources.append({'method': method, 'training_seed': training_seed, 'directory': str(directory.resolve()),
                            'metrics_sha256': hashlib.sha256((directory / 'eval_metrics.csv').read_bytes()).hexdigest()})
    for method, rows in grouped.items():
        entry = {'scenario': scene, 'algorithm': method, 'method': METHOD_LABELS.get(method, LABELS.get(method, method)), **endpoint_stats(rows)}
        for metric in METRICS:
            means = [r[f'{metric}_mean'] for r in per_seed if r['method'] == METHOD_LABELS.get(method)]
            entry[f'{metric}_training_seed_std'] = float(np.std(means, ddof=0)) if means else ''
        tables.append(entry)
    output = root / scene / 'summary'
    write_csv(output / 'table_iv.csv', tables); write_csv(output / 'per_training_seed.csv', per_seed)
    write_csv(output / 'v2_eval_metrics.csv', combined)
    versions = [{'scenario': scene, 'method': 'V1', 'update_epochs': 80, 'training_seeds': 1, **endpoint_stats(baselines['ours']['rows'])},
                *[{'scenario': scene, 'method': METHOD_LABELS[m], 'update_epochs': 10, 'training_seeds': 3, **endpoint_stats(grouped[m])} for m in METHODS]]
    write_csv(output / 'version_comparison.csv', versions)
    lines = [r'\begin{tabular}{lrrrrr}', r'Method & Runs & RMSE [nT] & Trr [\%] & RMSE train SD & Trr train SD \\', r'\hline']
    for row in tables:
        train_sd = ['--' if row[f'{k}_training_seed_std'] == '' else f"{row[f'{k}_training_seed_std']:.4f}" for k in ('rmse', 'trr')]
        lines.append(f"{row['method']} & {row['runs']} & ${row['rmse_mean']:.4f} \\pm {row['rmse_std']:.4f}$ & ${row['trr_mean']:.4f} \\pm {row['trr_std']:.4f}$ & " + ' & '.join(train_sd) + r' \\')
    lines.append(r'\end{tabular}')
    (output / 'table_iv.tex').write_text('\n'.join(lines) + '\n')
    write_json(output / 'sources.json', {'sources': sources, 'baseline_sources': {k: str(v['directory'].resolve()) for k, v in baselines.items()},
                'std': 'population, ddof=0', 'paired_test_seeds': '150 episodes share 50 starts across 3 training seeds; not 150 independent starts',
                'msr': 'Reused original single checkpoint and existing evaluation products', 'version_comparison': 'V1 to V2 is not a single-factor ablation'})
    import matplotlib.pyplot as plt
    for metric, ylabel in (('rmse', 'RMSE (nT)'), ('trr', 'Trr (%)')):
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, rows in grouped.items():
            steps, mean, std = _step_statistics(rows, rows[0]['algorithm'], metric)
            line, = ax.plot(steps, mean, label=METHOD_LABELS.get(method, LABELS.get(method, method)))
            ax.fill_between(steps, mean-std, mean+std, color=line.get_color(), alpha=.15)
        ax.set(xlabel='Step', ylabel=ylabel, xlim=(0, 70))
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        save_figure(fig, root / scene / 'figures' / metric, formats)
    return tables
