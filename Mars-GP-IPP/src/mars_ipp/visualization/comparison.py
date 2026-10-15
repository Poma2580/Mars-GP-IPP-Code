from collections import defaultdict
from pathlib import Path
from mars_ipp.utils.config import validate_output_path
from urllib.parse import quote
import numpy as np
import matplotlib.pyplot as plt
import yaml
from .learning_curves import read_csv, save_figure, plot_training
from .field_maps import plot_snapshot
from mars_ipp.evaluation.statistics import summarize, load_runs, ALGORITHMS, LABELS
from mars_ipp.utils.logger import write_csv


def _step_statistics(rows, algorithm, key):
    groups = defaultdict(list)
    for row in rows:
        if row['algorithm'] != algorithm:
            continue
        step = int(row['step'])
        if (row['stage'] == 'reset' and step == 0) or (row['stage'] == 'step' and step > 0):
            groups[step].append(float(row[key]))
    steps = sorted(groups)
    means = np.asarray([np.mean(groups[step]) for step in steps])
    stds = np.asarray([np.std(groups[step], ddof=0) for step in steps])
    return steps, means, stds


def _plot_evaluation(selected, output, formats):
    algorithms = sorted({row['algorithm'] for row in selected})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in zip(axes, ('rmse', 'trace_current'), ('RMSE (nT)', 'GP Trace (nT²)')):
        for algorithm in algorithms:
            steps, mean, std = _step_statistics(selected, algorithm, key)
            label_algorithm = LABELS.get(algorithm, algorithm)
            line, = ax.plot(steps, mean, label=label_algorithm, linewidth=1.5)
            ax.fill_between(steps, mean-std, mean+std, color=line.get_color(), alpha=.18)
        ax.set(xlabel='Step', ylabel=label)
        ax.grid(alpha=.5)
        ax.legend(fontsize=9)
    save_figure(fig, output / 'evaluation', formats)

    for algorithm in algorithms:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for key, label, color in (('rmse_Br', 'Br error', 'teal'),
                                  ('rmse_B2', 'Bφ error', 'red'),
                                  ('rmse_B3', 'Bψ error', 'purple')):
            steps, mean, _ = _step_statistics(selected, algorithm, key)
            ax.plot(steps, mean, color=color, linestyle='--', linewidth=1.5, label=label)
        ax.set(xlabel='Step', ylabel='RMSE (nT)')
        ax.grid(alpha=.6)
        ax.legend(loc='upper right')
        filename = 'component_rmse'
        if len(algorithms) > 1:
            ax.set_title(algorithm)
            filename += '_' + quote(algorithm, safe='-_')
        save_figure(fig, output / filename, formats)


def plot_results(directory, output=None, formats=('png',)):
    directory = Path(directory)
    output = validate_output_path(output if output else directory / 'figures')
    output.mkdir(parents=True, exist_ok=True)
    if (directory / 'comparison.json').exists():
        import json
        manifest = json.loads((directory / 'comparison.json').read_text())
        runs = load_runs([directory / name for name in manifest['algorithms']])
        plot_comparison(runs, output, formats)
        return
    if not (directory / 'eval_metrics.csv').exists():
        if not (directory / 'train_metrics.csv').exists():
            raise ValueError('No training or formal evaluation metrics in this directory')
        plot_training(directory, output, formats)
        return
    rows = read_csv(directory / 'eval_metrics.csv')
    latest = {a: max(int(r['episode']) for r in rows if r['algorithm'] == a) for a in {r['algorithm'] for r in rows}}
    selected = [r for r in rows if int(r['episode']) == latest[r['algorithm']]]
    _plot_evaluation(selected, output, formats)
    summary = summarize(rows)
    write_csv(output / 'summary.csv', summary)
    lines = [r'\begin{tabular}{lrr}', r'Method & RMSE (nT) & Trace (nT$^2$) \\', r'\hline']
    for row in summary:
        label = LABELS.get(row['algorithm'], row['algorithm']).replace('_', r'\_')
        lines.append(f"{label} & ${row['rmse_mean']:.4f} \\pm {row['rmse_std']:.4f}$ & ${row['trace_current_mean']:.4f} \\pm {row['trace_current_std']:.4f}$ " + r'\\')
    lines.append(r'\end{tabular}')
    (output / 'summary.tex').write_text('\n'.join(lines) + '\n')
    plot_training(directory, output, formats)
    snapshot = directory / 'field_snapshot.npz'
    if snapshot.exists():
        config = yaml.safe_load((directory / 'config.yaml').read_text())
        plot_snapshot(snapshot, output / 'field_map', formats,
                      scenario=config['field']['test_scenario'])


def plot_comparison(runs, output, formats=('png', 'pdf'), methods=None):
    output = validate_output_path(output)
    for key, label in (('trr', 'Trr (%)'), ('rmse', 'RMSE (nT)')):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name in (methods if methods is not None else (*ALGORITHMS, *(['ours'] if 'ours' in runs else []))):
            steps, mean, std = _step_statistics(runs[name]['rows'], runs[name].get('algorithm', name), key)
            line, = ax.plot(steps, mean, label=LABELS.get(name, name))
            if name != 'cover' or runs[name]['config']['evaluation'].get('cover_random_start'):
                ax.fill_between(steps, mean-std, mean+std, color=line.get_color(), alpha=.18)
        ax.set(xlabel='Step', ylabel=label, xlim=(0, 70))
        ax.grid(alpha=.3)
        ax.legend(fontsize=9)
        save_figure(fig, output / key, formats)
