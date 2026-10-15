"""All figures read saved results; no training or model inference in plotting."""
import csv
from pathlib import Path
from mars_ipp.utils.config import validate_output_path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def save_figure(fig, prefix, formats=('png',)):
    prefix = validate_output_path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in formats:
        fig.savefig(prefix.with_suffix('.' + suffix), dpi=160)
    plt.close(fig)


def _training_blocks(rows):
    """Summarize consecutive blocks of 100 recorded training episodes."""
    keys = ('episode', 'reward', *[key for key in rows[0] if key.startswith('r_')])
    values = np.asarray([[float(row[key]) for key in keys] for row in rows])
    blocks = [values[start:start + 100] for start in range(0, len(rows), 100)]
    means = np.asarray([block.mean(axis=0) for block in blocks]).reshape(-1, len(keys))
    stds = np.asarray([block.std(axis=0) for block in blocks]).reshape(-1, len(keys))
    return means, stds


def plot_training(directory, output, formats=('png',)):
    path = Path(directory) / 'train_metrics.csv'
    if not path.exists():
        return
    rows = read_csv(path)
    config_path = Path(directory) / 'config.yaml'
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    reward_mode = config.get('reward', {}).get('mode')
    raw_trace = reward_mode == 'gp_trace_global_raw'
    raw_local_trace = reward_mode == 'gp_trace_local_raw'
    fig, axes = plt.subplots(2, 1, figsize=(8, 7))
    means, stds = _training_blocks(rows)
    episodes, rewards = means[:, 0], means[:, 1]
    axes[0].plot(episodes, rewards, label='Mean (100-episode blocks)')
    axes[0].fill_between(episodes, rewards - stds[:, 1], rewards + stds[:, 1],
                         alpha=.2, label='±1 SD across training episodes within each block')
    axes[0].set(xlabel='Training episode', ylabel='Episode reward (nT²)' if raw_trace or raw_local_trace else 'Episode reward')
    axes[0].legend(fontsize=8)
    for column, name in enumerate([key[2:] for key in rows[0] if key.startswith('r_')], start=2):
        axes[1].plot(episodes, means[:, column], label=name)
    axes[1].set(xlabel='Training episode', ylabel='Raw global Trace decrease (nT²)' if raw_trace else 'Raw local Trace decrease (nT²)' if raw_local_trace else 'Sum of unweighted normalized terms',
                title='Reward terms: mean over 100-episode blocks')
    axes[1].legend()
    save_figure(fig, Path(output) / 'training', formats)
    if raw_trace:
        updates = read_csv(Path(directory) / 'updates.csv')
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot([int(r['steps']) for r in updates], [float(r['value_loss']) for r in updates])
        axes[0].set(xlabel='Training steps', ylabel='Value loss (raw targets)')
        axes[1].plot([int(r['episode']) for r in rows], [float(r['value_mean']) for r in rows])
        axes[1].set(xlabel='Training episode', ylabel='Mean predicted value (nT²)')
        save_figure(fig, Path(output) / 'critic_diagnostics', formats)
    if 'outside_actions' in rows[0]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, key in zip(axes.flat, ('rmse', 'trace_current', 'repeated_visits', 'outside_actions')):
            values = np.array([float(row[key]) for row in rows])
            blocks = [values[i:i+100] for i in range(0, len(values), 100)]
            mean = np.array([b.mean() for b in blocks]); std = np.array([b.std(ddof=0) for b in blocks])
            ax.plot(episodes, mean); ax.fill_between(episodes, mean-std, mean+std, alpha=.2)
            ax.set(xlabel='Training episode', ylabel=key)
        save_figure(fig, Path(output) / 'training_metrics', formats)
    endpoint_path = Path(directory) / 'eval_endpoints.csv'
    # Existing training results remain plottable.
    if not endpoint_path.exists():
        endpoint_path = Path(directory) / 'eval_metrics.csv'
    if not endpoint_path.exists():
        return
    eval_rows = read_csv(endpoint_path)
    final_step = max(int(row['step']) for row in eval_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = sorted({int(row['episode']) for row in eval_rows})
    for ax, key, label in zip(axes, ('rmse', 'trace_current'), ('RMSE (nT)', 'GP Trace (nT²)')):
        groups = [[float(row[key]) for row in eval_rows if int(row['episode']) == ep and row['stage'] == 'step' and int(row['step']) == final_step] for ep in epochs]
        means, stds = np.asarray([np.mean(g) for g in groups]), np.asarray([np.std(g) for g in groups])
        ax.plot(epochs, means)
        ax.fill_between(epochs, means-stds, means+stds, alpha=.2)
        ax.set(xlabel='Training episode (fixed evaluation seeds)', ylabel=label)
    save_figure(fig, Path(output) / 'training_evaluation', formats)
