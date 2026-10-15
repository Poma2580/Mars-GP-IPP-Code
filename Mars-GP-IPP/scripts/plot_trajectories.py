"""Plot each method and a paired trajectory panel from saved CSVs (no GP rerun)."""
import _bootstrap
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml
from mars_ipp.envs.actions import ACTIONS
from mars_ipp.field.dataset import SCENARIOS
from mars_ipp.evaluation.statistics import LABELS
from mars_ipp.visualization.learning_curves import read_csv, save_figure
from mars_ipp.visualization.field_maps import _field_axes


def load_trajectory(directory, seed):
    config = yaml.safe_load((directory/'config.yaml').read_text())
    rows = [r for r in read_csv(directory/'trajectories.csv') if int(r['seed']) == seed]
    reset = [r for r in read_csv(directory/'eval_metrics.csv') if int(r['seed']) == seed and r['stage'] == 'reset']
    if len(reset) != 1 or [int(r['step']) for r in rows] != list(range(1, config['env']['max_steps'] + 1)):
        raise ValueError(f'Missing or incomplete trajectory for seed {seed}: {directory}')
    start = [int(reset[0]['start_i']), int(reset[0]['start_j'])]
    grid = np.asarray([start, *[(int(r['i']), int(r['j'])) for r in rows]])
    size = config['env']['grid_size']
    if np.any(grid < 0) or np.any(grid >= size):
        raise ValueError('Saved trajectory leaves the action grid')
    for previous, current, row in zip(grid[:-1], grid[1:], rows):
        candidate = previous + ACTIONS[int(row['action'])]
        expected = previous if np.any(candidate < 0) or np.any(candidate >= size) else candidate
        if not np.array_equal(current, expected):
            raise ValueError('Saved position differs from recorded action')
    scenario = config['field']['test_scenario']
    origin = np.asarray(SCENARIOS[scenario]['origin'])
    return grid * config['env']['coordinate_scale'] + origin, grid, config, reset[0]


def draw(ax, path, grid, config, algorithm, annotate_visits=False):
    origin = np.asarray(SCENARIOS[config['field']['test_scenario']]['origin'])
    extent = (config['env']['grid_size'] - 1) * config['env']['coordinate_scale']
    _field_axes(ax, np.array([origin[0], origin[0] + extent]), np.array([origin[1], origin[1] + extent]))
    ax.plot(path[:,0], path[:,1], '-o', color='tab:red', linewidth=1.1, markersize=2.2, label='Trajectory')
    ax.scatter(*path[0], s=65, c='limegreen', marker='o', edgecolors='black', zorder=5, label='Start')
    ax.scatter(*path[-1], s=65, c='gold', marker='s', edgecolors='black', zorder=6, label='End')
    if annotate_visits:
        positions, counts = np.unique(grid, axis=0, return_counts=True)
        for position, count in zip(positions, counts):
            if count > 1:
                xy = position * config['env']['coordinate_scale'] + origin
                ax.annotate(str(count), xy, xytext=(3, 3), textcoords='offset points', fontsize=7, color='blue')
    unique = len(np.unique(grid, axis=0))
    ax.set_title(f'{LABELS.get(algorithm, algorithm)}\n{unique} unique positions', fontsize=10)


def run(directory, seed=1000, formats=('png','pdf'), methods=None):
    directory = Path(directory)
    algorithms = methods if methods is not None else ['gp_greedy','gp_trace','msr','cover','ours']
    loaded = [(a, *load_trajectory(directory/a, seed)) for a in algorithms]
    starts = {(int(item[-1]['start_i']), int(item[-1]['start_j'])) for item in loaded}
    scenarios = {item[3]['field']['test_scenario'] for item in loaded}
    if len(starts) != 1 or len(scenarios) != 1:
        raise ValueError('Comparison requires matching scenarios and actual initial positions')
    fig, axes = plt.subplots(1, len(algorithms), figsize=(22,5))
    for ax, (algorithm, path, grid, config, _) in zip(axes, loaded):
        draw(ax, path, grid, config, algorithm, annotate_visits=methods is not None)
        single, single_ax = plt.subplots(figsize=(6,6))
        draw(single_ax, path, grid, config, algorithm, annotate_visits=methods is not None)
        single_ax.legend(loc='best', fontsize=8)
        single.suptitle(f"{config['field']['test_scenario'].title()} — seed {seed}")
        save_figure(single, directory/algorithm/'figures'/f'trajectory_seed_{seed}', formats)
    axes[-1].legend(loc='best', fontsize=8)
    fig.suptitle(f'{next(iter(scenarios)).title()} — paired trajectories, seed {seed}, 70 actions')
    prefix = directory/'figures'/f'trajectories_seed_{seed}'
    save_figure(fig, prefix, formats)
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='One scenario comparison directory containing all five methods')
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--formats', nargs='+', choices=('png','pdf'), default=['png','pdf'])
    args = parser.parse_args()
    print(run(args.directory, args.seed, args.formats))
