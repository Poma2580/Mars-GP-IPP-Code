"""Descriptive statistics: population SD across fixed test runs, not variance."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import yaml
import numpy as np
from mars_ipp.utils.config import validate_output_path
from mars_ipp.utils.logger import write_csv, write_json


def summarize(rows):
    latest = {a: max(int(r['episode']) for r in rows if r['algorithm'] == a) for a in {r['algorithm'] for r in rows}}
    final_step = max(int(r['step']) for r in rows if r['stage'] == 'step')
    groups = defaultdict(list)
    for row in rows:
        if int(row['episode']) == latest[row['algorithm']] and int(row['step']) == final_step and row['stage'] == 'step':
            groups[row['algorithm']].append(row)
    result = []
    for algorithm, group in groups.items():
        entry = {'algorithm': algorithm, 'runs': len(group), 'episode': latest[algorithm], 'step': final_step}
        for key in ('rmse', 'trace_current', 'rmse_Br', 'rmse_B2', 'rmse_B3', 'trace_prior', 'trr'):
            if key not in group[0]:
                continue
            values = np.asarray([float(row[key]) for row in group])
            entry[f'{key}_mean'] = float(values.mean())
            entry[f'{key}_std'] = float(values.std(ddof=0))
        result.append(entry)
    return result


ALGORITHMS = ('gp_greedy', 'gp_trace', 'msr', 'cover')
LABELS = dict(zip(ALGORITHMS, ('GP-Entropy Active Learning', 'GP-Trace Active Learning', 'Msr', 'Cover')))
LABELS['ours'] = 'Ours'


def load_runs(directories):
    from mars_ipp.visualization.learning_curves import read_csv
    runs = {}
    for directory in map(Path, directories):
        config = yaml.safe_load((directory / 'config.yaml').read_text())
        if config['env']['grid_size'] != 16 or config['env']['max_steps'] != 70 or config['field']['test_scenario'] not in ('zhurong', 'perseverance'):
            raise ValueError('Expected grid16, 70 steps, supported Mars scene')
        rows = read_csv(directory / 'eval_metrics.csv')
        names = {r['algorithm'] for r in rows}
        if len(names) != 1 or not names <= set((*ALGORITHMS, 'ours')):
            raise ValueError('Each input must contain one requested baseline')
        name = names.pop()
        if name in runs:
            raise ValueError(f'Duplicate algorithm: {name}')
        if len({int(r['episode']) for r in rows}) != 1:
            raise ValueError('Mixed checkpoint episodes')
        seeds = sorted({int(r['seed']) for r in rows})
        expected = config['evaluation']['seeds'][:1] if name == 'cover' and not config['evaluation'].get('cover_random_start') else config['evaluation']['seeds']
        if seeds != sorted(expected):
            raise ValueError(f'Unexpected seeds: {name}')
        for seed in seeds:
            selected = [r for r in rows if int(r['seed']) == seed]
            stages = [(r['stage'], int(r['step'])) for r in selected]
            if stages != [('prior', 0), ('reset', 0)] + [('step', i) for i in range(1, 71)]:
                raise ValueError(f'Incomplete or duplicate stages: {name}, {seed}')
            if len({int(r['episode']) for r in selected}) != 1:
                raise ValueError('Mixed checkpoint episodes')
            prior = float(selected[0]['trace_prior'])
            if prior <= 0:
                raise ValueError('Invalid prior Trace')
            for row in selected:
                values = [float(row[k]) for k in ('rmse', 'trace_prior', 'trace_current', 'trr')]
                if not np.isfinite(values).all() or values[0] < 0 or values[2] < 0:
                    raise ValueError('Invalid metrics')
                if not np.isclose(values[1], prior) or not np.isclose(values[3], values[2] / prior * 100):
                    raise ValueError('Inconsistent Trr')
        trajectories = read_csv(directory / 'trajectories.csv')
        for seed in seeds:
            path = [r for r in trajectories if int(r['seed']) == seed]
            if [int(r['step']) for r in path] != list(range(1, 71)):
                raise ValueError('Incomplete trajectory')
        runs[name] = {'rows': rows, 'trajectories': trajectories, 'directory': directory, 'config': config}
    if not set(ALGORITHMS) <= set(runs) or not set(runs) <= set((*ALGORITHMS, 'ours')):
        raise ValueError('Supply exactly the four baseline evaluation directories')
    names = list(ALGORITHMS) + (['ours'] if 'ours' in runs else [])
    random_seeds = [{int(r['seed']) for r in runs[a]['rows']} for a in names if a != 'cover' or runs[a]['config']['evaluation'].get('cover_random_start')]
    if any(seeds != random_seeds[0] for seeds in random_seeds[1:]):
        raise ValueError('Random-start baselines must use identical seed sets')
    # Field parameters, sampling protocol and environment must agree across runs.
    metadata = [json.loads((runs[a]['directory'] / 'metadata.json').read_text()) for a in names]
    for name, meta in zip(names, metadata):
        run = runs[name]
        if (run['config']['env'] != runs[ALGORITHMS[0]]['config']['env'] or meta['fields'] != metadata[0]['fields']
                or run['config']['field']['test_scenario'] != runs[ALGORITHMS[0]]['config']['field']['test_scenario']):
            raise ValueError('Incompatible environment or field models')
    if runs['cover']['config']['evaluation'].get('cover_random_start'):
        def starts(run):
            return {int(r['seed']): (int(r['start_i']), int(r['start_j']))
                    for r in run['rows'] if r['stage'] == 'reset'}
        reference = starts(runs[ALGORITHMS[0]])
        if any(starts(runs[name]) != reference for name in names):
            raise ValueError('Random-start methods must have identical actual initial positions')
    return runs


def summarize_runs(runs, output):
    output = validate_output_path(output)
    table = []
    for name in (*ALGORITHMS, *(['ours'] if 'ours' in runs else [])):
        final = [r for r in runs[name]['rows'] if r['stage'] == 'step' and int(r['step']) == 70]
        entry = {'algorithm': name, 'method': LABELS[name], 'runs': len(final)}
        for key in ('trr', 'rmse'):
            values = np.asarray([float(r[key]) for r in final])
            for suffix, value in zip(('avg', 'std', 'max', 'min'), (values.mean(), values.std(ddof=0), values.max(), values.min())):
                entry[f'{key}_{suffix}'] = '—' if name == 'cover' and len(final) == 1 and suffix == 'std' else float(value)
        for key in ('unique_positions', 'repeated_visits', 'outside_actions', 'coverage'):
            if key in final[0]:
                values = np.asarray([float(r[key]) for r in final])
                entry[f'{key}_avg'] = float(values.mean())
                entry[f'{key}_std'] = float(values.std(ddof=0))
        table.append(entry)
    write_csv(output / 'table_iv.csv', table)
    lines = [r'\begin{tabular}{lrrrrrrrr}', r'\hline',
             r'Method & \multicolumn{4}{c}{Trr [\%]} & \multicolumn{4}{c}{RMSE [nT]} \\',
             r' & Avg & Std & Max & Min & Avg & Std & Max & Min \\', r'\hline']
    for row in table:
        values = [('--' if row[f'{k}_{s}'] == '—' else f"{row[f'{k}_{s}']:.4f}") for k in ('trr', 'rmse') for s in ('avg', 'std', 'max', 'min')]
        lines.append(LABELS[row['algorithm']] + ' & ' + ' & '.join(values) + r' \\')
    lines.extend([r'\hline', r'\end{tabular}'])
    (output / 'table_iv.tex').write_text('\n'.join(lines) + '\n')
    write_json(output / 'sources.json', {
        'protocol': ('All methods: paired seeded random starts; Cover uses diagonal serpentine boundary reflection. Msr reuses its old-protocol trained policy.'
                     if runs['cover']['config']['evaluation'].get('cover_random_start') else
                     'Cover: fixed start (14,14), one run. Others: seeded random starts. Not a fully paired experiment.'),
        'std': 'population standard deviation, ddof=0',
        'formal_seed_protocol': all({int(r['seed']) for r in runs[a]['rows']} == set(range(1000,1050)) for a in ALGORITHMS[:3]),
        'sources': {a: {'directory': str(runs[a]['directory'].resolve()),
                        'metrics_sha256': hashlib.sha256((runs[a]['directory'] / 'eval_metrics.csv').read_bytes()).hexdigest()} for a in runs}})
    return table

