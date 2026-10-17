"""Isolated reward-weight experiments, paired evaluation and artifact audits."""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch

from mars_ipp.envs.actions import ACTIONS
from mars_ipp.envs.observations import GPTraceScaler
from mars_ipp.evaluation.evaluator import evaluate
from mars_ipp.field.field_model import FieldModel
from mars_ipp.training.checkpoint import read_checkpoint
from mars_ipp.utils.config import ROOT, load_config
from mars_ipp.utils.logger import write_csv, write_json
from mars_ipp.visualization.learning_curves import read_csv, save_figure
from .v3_variants import TRAIN_MAPS, WEIGHT_EXPERIMENTS
from .v3_pipeline import (METRICS, empty_output, load_policy, prepare_data, protect_history,
                          scene_report, sha, train_variant, validate_scene, verify_hashes)

LOG = logging.getLogger(__name__)
GROUPS = ('v3_3', 'v3_4')
NOTE = ('Known per-map kernel and Bmax priors. Trr=Trace_current/Trace_prior*100%, lower is better. '
        'Population SD ddof=0. Reward components r_* are unweighted; weighted_r_* include coefficients. '
        'sum_reward retains the legacy cumulative unweighted GP component; reward_total is the step total.')


def names(group):
    return tuple(n for n, (v, _) in WEIGHT_EXPERIMENTS.items() if v == group)


@contextmanager
def experiment_log(name, stage):
    path = ROOT / f'logs/{name}_{stage}_{time.time_ns()}.log'
    path.parent.mkdir(exist_ok=True)
    handler = logging.FileHandler(path, mode='x')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(handler)
    try:
        yield path
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def group_root(root, group, smoke):
    return Path(root) / f'{group}_weight_{"smoke" if smoke else "train"}'


def check_new(paths):
    paths = [Path(p).resolve() for p in paths]
    if len(set(paths)) != len(paths) or any(a in b.parents for a in paths for b in paths if a != b):
        raise ValueError('Output directories must be distinct and non-nested')
    for path in paths:
        if path.exists():
            raise FileExistsError(f'Refuse to overwrite existing directory: {path}')


def preflight(train_root, test_root, smoke=False):
    configs = {}
    for name, (version, weight) in WEIGHT_EXPERIMENTS.items():
        config = load_config(ROOT / f'configs/main/{name}.yaml')
        base = load_config(ROOT / f'configs/main/{version}.yaml')
        expected = deepcopy(base)
        expected['experiment'] = name
        expected['reward']['weights'] = [1., weight]
        if config != expected:
            raise ValueError(f'Only experiment name and auxiliary weight may differ: {name}')
        configs[name] = config
    data = prepare_data(copy=False)
    for item in data.values():
        field = FieldModel.load(Path(item['target']) if Path(item['target']).exists() else Path(item['source']))
        if not np.isfinite(field.truth(np.array([[0., 0.], [30., 30.]]))).all():
            raise ValueError('Nonfinite field')
    if not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA device is unavailable')
    check_new([group_root(root, group, smoke) for root in (train_root, test_root) for group in GROUPS])
    return {'cuda': torch.cuda.get_device_name(), 'data': data,
            'configs': configs, 'smoke': smoke, 'note': NOTE}


def plot_training(directory, label, output=None):
    import matplotlib.pyplot as plt
    directory = Path(directory)
    target = Path(output or directory)
    rows = read_csv(directory / 'others/train_metrics.csv')
    component_names = [k[2:] for k in rows[0] if k.startswith('r_')]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    # Keep the established 100-episode blocks; smoke uses individual episodes.
    block_size = 1 if len(rows) < 100 else 100
    blocks = [rows[i:i + block_size] for i in range(0, len(rows), block_size)]
    x = [np.mean([float(r['episode']) for r in block]) for block in blocks]
    def series(key):
        values = [[float(r[key]) for r in block] for block in blocks]
        return np.array([np.mean(v) for v in values]), np.array([np.std(v, ddof=0) for v in values])
    mean, std = series('reward')
    axes[0].plot(x, mean, label='total')
    axes[0].fill_between(x, mean - std, mean + std, alpha=.2, label='±1 population SD')
    for name in component_names:
        axes[1].plot(x, series(f'r_{name}')[0], label=name)
        axes[2].plot(x, series(f'weighted_r_{name}')[0], label=name)
    for ax, title in zip(axes, ('Total reward', 'Unweighted components', 'Weighted components')):
        ax.set(xlabel='Training episode', ylabel='Reward (dimensionless)', title=title)
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle(label)
    save_figure(fig, target / 'figures/training')
    monitor = read_csv(directory / 'others/eval_endpoints.csv')
    epochs = sorted({int(r['episode']) for r in monitor})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, key in zip(axes, ('rmse', 'trace_current', 'trr')):
        for scene in TRAIN_MAPS:
            ax.plot(epochs, [np.mean([float(r[key]) for r in monitor if int(r['episode']) == ep
                                     and r['map'] == scene]) for ep in epochs], alpha=.55, label=scene)
        ax.plot(epochs, [np.mean([float(r[key]) for r in monitor if int(r['episode']) == ep])
                         for ep in epochs], color='black', linewidth=2, label='equal-map mean')
        ax.set(xlabel='Training episode', ylabel=key); ax.legend(fontsize=7)
    fig.suptitle(label)
    save_figure(fig, target / 'figures/training_evaluation')


def audit_training(directory, name, smoke):
    directory = Path(directory)
    rows = read_csv(directory / 'others/train_metrics.csv')
    updates = read_csv(directory / 'others/updates.csv')
    status = json.loads((directory / 'others/run_status.json').read_text())
    payload = read_checkpoint(directory / 'checkpoint.pt')
    episodes = 30 if smoke else 10000
    if payload['config']['experiment'] != name or payload['config']['reward']['weights'] != [1., WEIGHT_EXPERIMENTS[name][1]]:
        raise ValueError('Checkpoint experiment/weight mismatch')
    if payload['progress'] != {'episode': episodes, 'steps': episodes * 70, 'completed': True}:
        raise ValueError('Incomplete training checkpoint')
    if status['smoke'] != smoke or not status['completed'] or status['pending_rollout'] != 0:
        raise ValueError('Incomplete training status')
    if [int(r['episode']) for r in rows] != list(range(1, episodes + 1)):
        raise ValueError('Missing training episodes')
    if [r['map'] for r in rows] != [TRAIN_MAPS[i % 5] for i in range(episodes)]:
        raise ValueError('Incorrect map rotation')
    if Counter(r['map'] for r in rows) != Counter(dict.fromkeys(TRAIN_MAPS, episodes // 5)):
        raise ValueError('Unbalanced maps')
    if [int(r['samples']) for r in updates] != ([2000, 100] if smoke else [2000] * 350):
        raise ValueError('Unexpected rollout update sizes')
    component = 'rep' if WEIGHT_EXPERIMENTS[name][0] == 'v3_3' else 'mh'
    for row in rows:
        if not np.isfinite([float(v) for k, v in row.items() if k != 'map']).all():
            raise ValueError('Nonfinite training metric')
        gp, aux = float(row['r_gp']), float(row[f'r_{component}'])
        np.testing.assert_allclose([float(row['weighted_r_gp']), float(row[f'weighted_r_{component}']),
                                   float(row['reward'])], [gp, aux * WEIGHT_EXPERIMENTS[name][1],
                                                         gp + aux * WEIGHT_EXPERIMENTS[name][1]], atol=1e-10)
        np.testing.assert_allclose(gp, float(row['raw_trace_sum']) / float(row['trace_prior']), atol=1e-7)
        np.testing.assert_allclose(float(row['trr']), 100 * float(row['trace_current']) / float(row['trace_prior']))
        if int(row['unique_positions']) + int(row['repeated_visits']) + int(row['outside_actions']) != 71:
            raise ValueError('Invalid training action counts')
    for image in ('training.png', 'training_evaluation.png'):
        if not (directory / 'figures' / image).is_file():
            raise ValueError('Missing training figure')
    result = {'experiment': name, 'smoke': smoke, 'episodes': episodes, 'steps': episodes * 70,
              'map_counts': status['map_counts'], 'update_samples': [int(r['samples']) for r in updates],
              'checkpoint_sha256': sha(directory / 'checkpoint.pt'), 'passed': True}
    write_json(directory / 'others/acceptance.json', result)
    return result


def require_formal(directory, name):
    payload = read_checkpoint(Path(directory) / 'checkpoint.pt')
    status = json.loads((Path(directory) / 'others/run_status.json').read_text())
    if (payload['progress'] != {'episode': 10000, 'steps': 700000, 'completed': True}
            or payload['config']['training']['episodes'] != 10000 or status.get('smoke') is not False
            or payload['config']['experiment'] != name or not status.get('completed')):
        raise ValueError(f'Formal evaluation requires completed non-smoke training: {name}')


def audit_scene(runs, paths, seeds, methods, configs):
    """Replay saved actions independently of the environment and report builder."""
    validate_scene(runs, paths, seeds, methods)
    for method in methods:
        for seed in seeds:
            rows = [r for r in runs[method] if int(r['seed']) == seed]
            trajectory = [r for r in paths[method] if int(r['seed']) == seed]
            position = np.array([int(rows[1]['start_i']), int(rows[1]['start_j'])])
            visited = {tuple(position)}; repeated = outside = 0
            prior = float(rows[0]['trace_prior'])
            for row, action_row in zip(rows[2:], trajectory):
                action = int(action_row['action'])
                if not 0 <= action < len(ACTIONS):
                    raise ValueError('Invalid saved action')
                nxt = position + ACTIONS[action]
                is_outside = bool(np.any(nxt < 0) or np.any(nxt >= 16))
                is_repeat = not is_outside and tuple(nxt) in visited
                sampled = not is_outside and not is_repeat
                if is_outside:
                    outside += 1
                else:
                    repeated += int(is_repeat); position = nxt; visited.add(tuple(position))
                if [int(action_row['i']), int(action_row['j'])] != position.tolist():
                    raise ValueError('Saved action/position mismatch')
                if [int(row[k]) for k in ('unique_positions', 'repeated_visits', 'outside_actions')] != [len(visited), repeated, outside]:
                    raise ValueError('Saved action counts differ from replay')
                if int(row['samples']) != len(visited) or str(row['sampled']).lower() != str(sampled).lower():
                    raise ValueError('Saved sampling flag/count mismatch')
                if float(row['trace_prior']) != prior:
                    raise ValueError('Trace prior changed within an episode')
                if method != 'gp_trace':
                    config = configs[method]
                    extra = 'rep' if config['version'] == 'v3_3' else 'mh'
                    gp, aux = float(row['r_gp']), float(row[f'r_{extra}'])
                    expected = gp + config['reward']['weights'][1] * aux
                    np.testing.assert_allclose([float(row['reward_total']), float(action_row['reward']),
                                               float(row['weighted_r_gp']) + float(row[f'weighted_r_{extra}'])],
                                              [expected] * 3, atol=1e-12)
                    if extra == 'rep' and aux != -float(is_repeat):
                        raise ValueError('Repeat reward sign mismatch')
                    if not sampled and (gp != 0 or (extra == 'mh' and aux != 0)):
                        raise ValueError('Unsampled action earned GP/MH reward')
                    if is_outside and expected != 0:
                        raise ValueError('Outside reward must be zero')


def write_report(output, smoke):
    output = Path(output)
    rows = [dict(r, map=f'test{i}') for i in range(1, 6) for r in read_csv(output / f'others/test{i}/summary.csv')
            if int(r['step']) == 70]
    write_csv(output / 'others/summary_70.csv', rows)
    lines = ['# 权重实验：' + ('冒烟验收，非正式性能结果' if smoke else '正式测试'), '', NOTE, '',
             '| 地图 | 方法 | Trr (%) | RMSE (nT) | 重复 | 越界 |', '|---|---|---|---|---|---|']
    diagnostics = []
    for i in range(1, 6):
        scene = f'test{i}'
        for row in read_csv(output / f'others/{scene}/summary.csv'):
            if int(row['step']) != 70:
                continue
            cells = [f"{float(row[k+'_mean']):.4f} ± {float(row[k+'_std']):.4f}"
                     for k in ('trr', 'rmse', 'repeated_visits', 'outside_actions')]
            lines.append('| ' + ' | '.join([scene, row['method'], *cells]) + ' |')
            if row['method'] == 'gp_trace':
                continue
            records = read_csv(output / f"others/{scene}/{row['method']}/eval_metrics.csv")
            step_rows = [r for r in records if r['stage'] == 'step']
            extra = 'rep' if row['method'].startswith('V3_3_') else 'mh'
            gp = sum(abs(float(r['weighted_r_gp'])) for r in step_rows)
            aux = sum(abs(float(r[f'weighted_r_{extra}'])) for r in step_rows)
            diagnostics.append({'map': scene, 'method': row['method'], 'abs_weighted_gp': gp,
                                'abs_weighted_aux': aux, 'aux_exceeds_gp': aux > gp,
                                'outside_mean': float(row['outside_actions_mean'])})
    lines += ['', '## 行为与奖励分量', '',
              '逐地图记录辅助项绝对累计量是否超过GP项，以及越界次数。出现越界只能说明该行为存在，不能单凭次数认定策略有意逃避惩罚。', '']
    for d in diagnostics:
        lines.append(f"- {d['map']} / {d['method']}: |GP|={d['abs_weighted_gp']:.6g}, "
                     f"|辅助|={d['abs_weighted_aux']:.6g}, 辅助项超过GP项={d['aux_exceeds_gp']}, "
                     f"平均越界={d['outside_mean']:.2f}。")
    if smoke:
        lines += ['', '仅完成30回合训练和两个测试seed；本结果用于验证流程，不判断最优权重，也不据此断言正式训练的奖励主导或越界逃避。']
    (output / 'others/report.md').write_text('\n'.join(lines) + '\n')
    write_json(output / 'others/reward_diagnostics.json', diagnostics)


def audit_report(output, scene, runs, methods, seeds):
    """Recompute table statistics from raw rows, separate from report generation."""
    output = Path(output)
    summaries = read_csv(output / f'others/{scene}/summary.csv')
    if len(summaries) != len(methods) * 3:
        raise ValueError('Wrong summary row count')
    table = (output / f'{scene}/table/comparison.md').read_text()
    for budget in (30, 50, 70):
        section = table.split(f'## 预算{budget}步表现\n')[1].split('## ')[0]
        for method in methods:
            rows = [r for r in runs[method] if r['stage'] == 'step' and int(r['step']) == budget]
            if sorted(int(r['seed']) for r in rows) != sorted(seeds):
                raise ValueError('Wrong budget seeds')
            summary = next(r for r in summaries if int(r['step']) == budget and r['method'] == method)
            if int(summary['runs']) != len(seeds):
                raise ValueError('Wrong budget sample count')
            cells = []
            for key in METRICS:
                values = np.array([float(r[key]) for r in rows])
                mean, std = values.mean(), values.std(ddof=0)
                np.testing.assert_allclose([float(summary[key + '_mean']), float(summary[key + '_std'])],
                                           [mean, std], rtol=1e-12, atol=1e-12)
                cells.append(f'{mean:.4f} ± {std:.4f}')
            if '| ' + method + ' | ' + ' | '.join(cells) + ' |' not in section:
                raise ValueError('Markdown values differ from raw records')


def evaluate_groups(train_root, test_root, smoke=False):
    configs = {name: load_config(ROOT / f'configs/main/{name}.yaml') for name in WEIGHT_EXPERIMENTS}
    # Check every model and output before writing any evaluation artifacts.
    check_new([group_root(test_root, group, smoke) for group in GROUPS])
    for name, (version, _) in WEIGHT_EXPERIMENTS.items():
        directory = group_root(train_root, version, smoke) / name
        if not smoke:
            require_formal(directory, name)
        payload = read_checkpoint(directory / 'checkpoint.pt')
        expected = deepcopy(configs[name])
        if smoke:
            expected['training'].update(episodes=30, checkpoint_every=10, eval_every=10)
        if payload['config'] != expected or payload['progress'] != {
                'episode': 30 if smoke else 10000, 'steps': 2100 if smoke else 700000, 'completed': True}:
            raise ValueError(f'Unexpected checkpoint protocol: {name}')
    seeds = [1000, 1001] if smoke else list(range(1000, 1050))
    outputs = {g: empty_output(group_root(test_root, g, smoke)) for g in GROUPS}
    baseline_cfg = load_config(ROOT / 'configs/main/v3_1_gp_trace_global_raw_train1.yaml')
    baseline_source = ROOT / 'src/mars_ipp/agents/baselines/gp_trace.py'
    metadata = {g: {'smoke': smoke, 'seeds': seeds, 'note': NOTE, 'fields': {}, 'checkpoints': {},
                     'baseline_source': {'path': str(baseline_source), 'sha256': sha(baseline_source)},
                     'baseline_config': baseline_cfg} for g in GROUPS}
    for g in GROUPS:
        for name in names(g):
            path = group_root(train_root, g, smoke) / name / 'checkpoint.pt'
            metadata[g]['checkpoints'][name] = {'path': str(path.resolve()), 'sha256': sha(path)}
    for i in range(1, 6):
        scene = f'test{i}'; field = FieldModel.load(ROOT / f'data/models/{scene}.pkl')
        bc = deepcopy(baseline_cfg)
        bc['field']['test_scenario'] = scene; bc['evaluation']['seeds'] = seeds
        with experiment_log('gp_trace', 'weight_evaluate'):
            baseline_rows, baseline_paths, _ = evaluate(field, GPTraceScaler(), bc, algorithm='gp_trace')
            LOG.info('Completed gp_trace %s: %d seeds', scene, len(seeds))
        source = outputs['v3_3'] / f'others/{scene}/gp_trace'
        for group, output in outputs.items():
            methods = ('gp_trace', *names(group))
            runs = {'gp_trace': deepcopy(baseline_rows)}; paths = {'gp_trace': deepcopy(baseline_paths)}
            metadata[group]['fields'][scene] = {'sha256': field.sha256, 'parameters': field.parameters}
            for name in names(group):
                agent, payload = load_policy(group_root(train_root, group, smoke) / name / 'checkpoint.pt')
                cfg = deepcopy(payload['config']); cfg['field']['test_scenario'] = scene
                cfg['evaluation']['seeds'] = seeds
                weights = {k: v.detach().clone() for k, v in agent.network.state_dict().items()}
                scaler = deepcopy(agent.scaler.to_dict())
                with experiment_log(name, 'evaluate'):
                    rows, trajectory, _ = evaluate(field, agent.scaler, cfg, agent, episode=payload['progress']['episode'])
                    LOG.info('Completed %s %s: %d seeds', name, scene, len(seeds))
                if scaler != agent.scaler.to_dict() or any(not weights[k].equal(v) for k, v in agent.network.state_dict().items()):
                    raise ValueError('Evaluation changed policy')
                for r in (*rows, *trajectory):
                    r['algorithm'] = name
                runs[name] = rows; paths[name] = trajectory
            for method in methods:
                base = output / f'others/{scene}/{method}'
                write_csv(base / 'eval_metrics.csv', runs[method])
                write_csv(base / 'trajectories.csv', paths[method])
            base = output / f'others/{scene}/gp_trace'
            for file in ('eval_metrics.csv', 'trajectories.csv'):
                if sha(base / file) != sha(source / file):
                    raise ValueError('Shared baseline result mismatch')
            write_json(base / 'provenance.json', {'source': str(source.resolve()), 'evaluated_once': True,
                       'sha256': {f: sha(base / f) for f in ('eval_metrics.csv', 'trajectories.csv')}})
            audit_scene(runs, paths, seeds, methods, configs)
            scene_report(scene, runs, paths, output / scene, seeds, smoke, methods=methods,
                         note=NOTE, aux_output=output / f'others/{scene}', weight_report=True)
            audit_report(output, scene, runs, methods, seeds)
            LOG.info('Completed %s %s: %d seeds', group, scene, len(seeds))
    for group, output in outputs.items():
        write_json(output / 'others/metadata.json', metadata[group])
        write_report(output, smoke)
        write_json(output / 'others/acceptance.json', {'passed': True, 'smoke': smoke,
                   'methods': ['gp_trace', *names(group)], 'maps': 5, 'seeds': seeds,
                   'paired_starts_and_replayed_actions': True, 'shared_baseline_verified': True})
    return outputs


def replot(train_root, test_root, output, smoke=False):
    """Rebuild figures/tables from CSV; never load a policy or a field."""
    check_new([output]); output = Path(output)
    for group in GROUPS:
        source = group_root(test_root, group, smoke)
        meta = json.loads((source / 'others/metadata.json').read_text())
        if meta['smoke'] != smoke:
            raise ValueError('Plot mode differs from source experiment')
        destination = group_root(output / 'comparison', group, smoke)
        methods = ('gp_trace', *names(group))
        for name in names(group):
            plot_training(group_root(train_root, group, smoke) / name,
                          'SMOKE — non-formal results' if smoke else name,
                          group_root(output / 'main', group, smoke) / name)
        for i in range(1, 6):
            scene = f'test{i}'; base = source / f'others/{scene}'
            runs = {m: read_csv(base / m / 'eval_metrics.csv') for m in methods}
            paths = {m: read_csv(base / m / 'trajectories.csv') for m in methods}
            scene_report(scene, runs, paths, destination / scene, meta['seeds'], smoke,
                         methods=methods, note=NOTE, aux_output=destination / f'others/{scene}', weight_report=True)
            for relative in (f'{scene}/table/comparison.md', f'others/{scene}/summary.csv'):
                if (source / relative).read_bytes() != (destination / relative).read_bytes():
                    raise ValueError(f'Replot differs from saved report: {relative}')
        write_json(destination / 'others/metadata.json', meta)
    write_json(output / 'acceptance.json', {'passed': True, 'csv_only': True, 'tables_and_summaries_identical': True})


def run(stage, train_root, test_root, output=None, smoke_results=False):
    smoke = stage == 'smoke' or smoke_results
    if stage == 'preflight':
        report = preflight(train_root, test_root)
        print(f"Preflight passed: {len(report['configs'])} configurations, 10 maps, CUDA {report['cuda']}; no outputs created.")
        return
    if stage == 'plot' and output is None:
        raise ValueError('plot requires a new --output directory')
    if stage in ('train', 'smoke'):
        report = preflight(train_root, test_root, smoke)
    protected = protect_history()
    try:
        if stage in ('train', 'smoke'):
            prepare_data(copy=True)
            for name, (group, _) in WEIGHT_EXPERIMENTS.items():
                with experiment_log(name, stage):
                    directory = group_root(train_root, group, smoke) / name
                    train_variant(report['configs'][name], directory, smoke, plotter=plot_training)
                    audit_training(directory, name, smoke)
            for group in GROUPS:
                root = group_root(train_root, group, smoke)
                write_json(root / 'others/data_manifest.json', report['data'])
        if stage in ('smoke', 'evaluate'):
            evaluate_groups(train_root, test_root, smoke)
        if stage == 'plot':
            replot(train_root, test_root, output, smoke)
    except Exception as exc:
        LOG.exception('Weight experiment failed at stage %s: %s', stage, exc)
        # Only write inside newly created outputs, never into historical inputs.
        if stage in ('smoke', 'train', 'evaluate'):
            roots = [train_root, test_root] if stage == 'smoke' else [train_root if stage == 'train' else test_root]
            for root in roots:
                for group in GROUPS:
                    path = group_root(root, group, smoke)
                    if path.exists() and not any(Path(p).is_relative_to(path.resolve()) for p in protected):
                        write_json(path / 'others/failure.json', {'stage': stage, 'error': repr(exc)})
        raise
    finally:
        verify_hashes(protected)
    roots = ([group_root(train_root, g, smoke) for g in GROUPS] if stage in ('train', 'smoke') else [])
    roots += [group_root(test_root, g, smoke) for g in GROUPS] if stage in ('smoke', 'evaluate') else []
    for root in roots:
        write_json(root / 'others/protected_inputs.json', protected)
        write_json(root / 'others/stage_acceptance.json', {'stage': stage, 'passed': True, 'smoke': smoke,
                   'protected_files_unchanged': len(protected), 'formal_training_started': stage == 'train'})
