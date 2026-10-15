"""Paired Full/No-B diagnostics from saved final-checkpoint evaluations."""
from collections import defaultdict
from pathlib import Path
from mars_ipp.utils.config import validate_output_path
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from mars_ipp.envs.actions import ACTIONS
from mars_ipp.training.checkpoint import read_checkpoint, scalers_equivalent
from mars_ipp.utils.config import merge, resolve_path
from mars_ipp.utils.logger import write_csv
from mars_ipp.visualization.learning_curves import read_csv, save_figure


def seed_statistics(metrics, trajectory, size=16):
    """Count legal revisits separately from rejected boundary actions."""
    stages = {(row['stage'], int(row['step'])): row for row in metrics}
    if len(metrics) != 72 or len(stages) != 72 or set(stages) != {
            ('prior', 0), ('reset', 0), *(('step', step) for step in range(1, 71))}:
        raise ValueError('Expected prior, reset and 70 unique metric steps')
    path = sorted(trajectory, key=lambda row: int(row['step']))
    if [int(row['step']) for row in path] != list(range(1, 71)):
        raise ValueError('Expected 70 unique trajectory steps')
    seed = int(metrics[0]['seed'])
    position = np.random.default_rng(seed).integers(0, size, 2)
    start = tuple(position)
    visited = {start}
    repeated = outside_count = 0
    counts = np.zeros(8, dtype=int)
    for row in path:
        action = int(row['action'])
        if not 0 <= action < 8:
            raise ValueError('Invalid action')
        counts[action] += 1
        candidate = position + ACTIONS[action]
        outside = bool(np.any(candidate < 0) or np.any(candidate >= size))
        expected = position if outside else candidate
        recorded = np.array([int(row['i']), int(row['j'])])
        info = stages[('step', int(row['step']))]
        if not np.array_equal(recorded, expected) or (str(info['sampled']).lower() == 'true') != (not outside):
            raise ValueError(f'Trajectory/sample mismatch for seed {seed}, step {row["step"]}')
        outside_count += outside
        if not outside:
            repeated += tuple(recorded) in visited
            visited.add(tuple(recorded))
        position = recorded
    result = {'seed': seed, 'start_i': start[0], 'start_j': start[1],
              'unique_cells': len(visited), 'repeat_visits': repeated, 'outside_actions': outside_count}
    for key, label in (('rmse', 'rmse'), ('trace_current', 'trace')):
        initial = float(stages[('reset', 0)][key])
        final = float(stages[('step', 70)][key])
        if initial <= 0:
            raise ValueError('Relative decline requires a positive initial metric')
        result.update({f'prior_{label}': float(stages[('prior', 0)][key]),
                       f'initial_{label}': initial, f'final_{label}': final,
                       f'{label}_decline_pct': 100 * (initial - final) / initial})
    result.update({f'action_{action}_pct': 100 * int(count) / 70 for action, count in enumerate(counts)})
    return result


def load_evaluation(directory, scenario, mode):
    directory = Path(directory)
    config = yaml.safe_load((directory / 'config.yaml').read_text())
    metadata = json.loads((directory / 'metadata.json').read_text())
    if config['field']['test_scenario'] != scenario or config['ppo'].get('observation_mode', 'full') != mode:
        raise ValueError(f'Wrong scenario or observation mode: {directory}')
    if config['evaluation']['seeds'] != list(range(1000, 1050)) or config['env']['max_steps'] != 70:
        raise ValueError(f'Wrong evaluation protocol: {directory}')
    rows, paths = read_csv(directory / 'eval_metrics.csv'), read_csv(directory / 'trajectories.csv')
    if not np.isfinite([[float(row[key]) for key in ('rmse', 'trace_current')] for row in rows]).all():
        raise ValueError(f'Nonfinite step metrics: {directory}')
    groups, trajectories = defaultdict(list), defaultdict(list)
    for collection, target in ((rows, groups), (paths, trajectories)):
        for row in collection:
            if int(row['episode']) != 10000 or row['algorithm'] != 'ppo':
                raise ValueError(f'Expected final Episode 10000 PPO: {directory}')
            target[int(row['seed'])].append(row)
    expected = set(range(1000, 1050))
    if set(groups) != expected or set(trajectories) != expected:
        raise ValueError(f'Missing or unexpected evaluation seeds: {directory}')
    stats = [seed_statistics(groups[seed], trajectories[seed], config['env']['grid_size']) for seed in sorted(expected)]
    if not np.isfinite([[value for key, value in row.items()] for row in stats]).all():
        raise ValueError(f'Nonfinite diagnostics: {directory}')
    return {'config': config, 'metadata': metadata, 'rows': rows, 'paths': trajectories,
            'stats': stats, 'directory': directory}


def validate_training(directory, mode):
    directory = Path(directory)
    status = json.loads((directory / 'run_status.json').read_text())
    if (status['episode'], status['steps'], status['completed'], status['pending_rollout'], status['updates']) != (10000, 700000, True, 0, 350):
        raise ValueError(f'Training is not complete: {directory}')
    config = yaml.safe_load((directory / 'config.yaml').read_text())
    if config['ppo'].get('observation_mode', 'full') != mode or config['seed'] != 0:
        raise ValueError(f'Wrong training mode/seed: {directory}')
    payload = read_checkpoint(directory / 'checkpoint.pt')
    if payload['config'] != config or payload['progress'] != {'episode': 10000, 'steps': 700000, 'completed': True}:
        raise ValueError(f'Final checkpoint/config mismatch: {directory}')
    metadata = json.loads((directory / 'metadata.json').read_text())
    if payload['scaler'] != metadata['scaler']:
        raise ValueError(f'Checkpoint/metadata Scaler mismatch: {directory}')
    return config, metadata


def summarize_group(group):
    result = {}
    for key in group['stats'][0]:
        if key in ('seed', 'start_i', 'start_j'):
            continue
        values = np.asarray([row[key] for row in group['stats']], dtype=float)
        result[f'{key}_mean'] = float(values.mean())
        result[f'{key}_std'] = float(values.std(ddof=0))
    return result


def plot_diagnostics(groups, full_training, no_b_training, output, report_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for row, scenario in enumerate(('zhurong', 'perseverance')):
        for col, method in enumerate(('Full', 'No-B')):
            group = groups[(scenario, method)]
            path = sorted(group['paths'][1000], key=lambda item: int(item['step']))
            start = group['stats'][0]
            coordinates = np.asarray([[start['start_i'], start['start_j']]] +
                                     [[int(item['i']), int(item['j'])] for item in path])
            coordinates = coordinates * group['config']['env']['coordinate_scale']
            ax = axes[row, col]
            ax.plot(coordinates[:, 0], coordinates[:, 1], '.-', linewidth=1, color='firebrick')
            ax.scatter(*coordinates[0], marker='o', color='green', label='Start', zorder=3)
            ax.scatter(*coordinates[-1], marker='s', color='black', label='End', zorder=3)
            ax.set(title=f'{scenario} / {method} / seed 1000', xlabel='Local longitude (deg)',
                   ylabel='Local latitude (deg)', xlim=(-1, 29), ylim=(-1, 29), aspect='equal')
            ax.grid(alpha=.3)
            ax.legend()
    save_figure(fig, output / 'paired_trajectories_seed_1000', ('png', 'pdf'))
    for scenario in ('zhurong', 'perseverance'):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for method in ('Full', 'No-B'):
            rows = groups[(scenario, method)]['rows']
            for ax, key, label in zip(axes, ('rmse', 'trace_current'), ('RMSE (nT)', 'GP Trace (nT²)')):
                values = [[float(item[key]) for item in rows if int(item['step']) == step and
                           item['stage'] == ('reset' if step == 0 else 'step')] for step in range(71)]
                mean, std = np.mean(values, axis=1), np.std(values, axis=1, ddof=0)
                line, = ax.plot(range(71), mean, label=method)
                ax.fill_between(range(71), mean-std, mean+std, color=line.get_color(), alpha=.15)
                ax.set(xlabel='Step', ylabel=label, title=scenario)
                ax.grid(alpha=.3)
                ax.legend()
        save_figure(fig, output / f'{scenario}_step_metrics', ('png', 'pdf'))
    fig, ax = plt.subplots(figsize=(8, 4))
    for method, directory in (('Full', full_training), ('No-B', no_b_training)):
        rows = read_csv(Path(directory) / 'eval_endpoints.csv')
        episodes = list(range(200, 10001, 200))
        values = []
        if len(rows) != 2500:
            raise ValueError(f'Expected 50 evaluation periods × 50 seeds: {directory}')
        for episode in episodes:
            selected = [r for r in rows if int(r['episode']) == episode]
            if len(selected) != 50 or {int(r['seed']) for r in selected} != set(range(1000, 1050)):
                raise ValueError(f'Incomplete training evaluation: {directory}, {episode}')
            if any(r['stage'] != 'step' or int(r['step']) != 70 for r in selected):
                raise ValueError('Expected training evaluation endpoints at step 70')
            values.append([float(r['rmse']) for r in selected])
        mean, std = np.mean(values, axis=1), np.std(values, axis=1, ddof=0)
        line, = ax.plot(episodes, mean, label=method)
        ax.fill_between(episodes, mean-std, mean+std, color=line.get_color(), alpha=.15)
    ax.set(xlabel='Training episode', ylabel='Perseverance endpoint RMSE (nT)')
    ax.legend()
    ax.grid(alpha=.3)
    save_figure(fig, output / 'training_perseverance_rmse', ('png', 'pdf'))


def run_diagnostics(full_zhurong, full_perseverance, no_b_zhurong, no_b_perseverance,
                    full_training, no_b_training, output, report_path=None):
    output = validate_output_path(output)
    groups = {}
    for scenario, method, path in (
            ('zhurong', 'Full', full_zhurong), ('perseverance', 'Full', full_perseverance),
            ('zhurong', 'No-B', no_b_zhurong), ('perseverance', 'No-B', no_b_perseverance)):
        groups[(scenario, method)] = load_evaluation(path, scenario, 'full' if method == 'Full' else 'position_only')
    full_config, full_metadata = validate_training(full_training, 'full')
    no_b_config, no_b_metadata = validate_training(no_b_training, 'position_only')
    expected = merge(full_config, {'experiment': 'no_b_rbf', 'seed': 0, 'seeds': [0],
                                  'ppo': {'observation_mode': 'position_only'}})
    if no_b_config != expected:
        raise ValueError('No-B training differs from Full beyond the allowed overrides')
    summaries, per_seed = [], []
    for (scenario, method), group in groups.items():
        training_config = full_config if method == 'Full' else no_b_config
        training_path = full_training if method == 'Full' else no_b_training
        if group['config'] != merge(training_config, {'field': {'test_scenario': scenario}}):
            raise ValueError('Evaluation configuration differs from training beyond the test scenario')
        if resolve_path(group['metadata']['checkpoint']).resolve() != (Path(training_path) / 'checkpoint.pt').resolve():
            raise ValueError('Evaluation references a different checkpoint')
        own_metadata = full_metadata if method == 'Full' else no_b_metadata
        if group['metadata']['scaler'] != own_metadata['scaler'] or not scalers_equivalent(no_b_metadata['scaler'], full_metadata['scaler']):
            raise ValueError('Scaler mismatch')
        for field, info in full_metadata['fields'].items():
            for metadata in (group['metadata'], no_b_metadata):
                if any(metadata['fields'][field][key] != info[key] for key in ('sha256', 'parameters')):
                    raise ValueError('Field hash or GP parameter mismatch')
        summaries.append({'scenario': scenario, 'method': method, 'runs': 50, **summarize_group(group)})
        per_seed.extend({'scenario': scenario, 'method': method, **item} for item in group['stats'])
    for scenario in ('zhurong', 'perseverance'):
        for full, no_b in zip(groups[(scenario, 'Full')]['stats'], groups[(scenario, 'No-B')]['stats']):
            if any(full[key] != no_b[key] for key in ('seed', 'start_i', 'start_j', 'initial_rmse', 'initial_trace')):
                raise ValueError('Paired seeds do not share initial conditions')
    report_path = Path(report_path) if report_path else resolve_path('docs/no_b_diagnostic_report.md')
    names = ('diagnostic_summary.csv', 'diagnostic_per_seed.csv', 'diagnostic_figures')
    if any((output / name).exists() for name in names):
        raise FileExistsError('Diagnostic artifacts already exist; choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    figures = output / 'diagnostic_figures'
    plot_diagnostics(groups, full_training, no_b_training, figures)
    write_csv(output / 'diagnostic_summary.csv', summaries)
    write_csv(output / 'diagnostic_per_seed.csv', per_seed)
    write_report(output, groups, summaries, full_training, no_b_training, report_path)
    return summaries


def write_report(output, groups, summaries, full_training, no_b_training, report_path):
    def fmt(row, key):
        return f"{row[key + '_mean']:.4f} ± {row[key + '_std']:.4f}"

    lines = ['# No-B 状态消融诊断报告', '',
             '四组均为最终 Episode 10000，训练 seed=0；评估种子 1000～1049，每种子 70 步。',
             '统计使用总体标准差（ddof=0）；相对下降率逐种子计算后汇总。初始指标为首次采样后的 reset；prior 为采样前指标。', '',
             '## 配置与验收', '',
             'No-B 仅覆盖 experiment=no_b_rbf、seed=0、seeds=[0]、ppo.observation_mode=position_only。五维网络及原始 Scaler 保持不变。',
             '修改：PPO 统一 prepare_state；Checkpoint 模式校验；专用配置、测试与诊断脚本。未修改 GP、环境、奖励或网络。',
             '已校验训练完成（10000 Episode / 700000 步 / 350 更新 / 空 buffer）、配置差异、磁场哈希、GP 参数、Scaler、评估完整性及配对初始条件。',
             '单元测试、smoke 及正式回测的实际执行记录见 [验证记录](no_b_validation.md)。', '',
             '## 四组结果', '',
             '| 场景 | 策略 | 初始 RMSE (nT) | 终点 RMSE (nT) | RMSE 下降 (%) | 初始 Trace (nT²) | 终点 Trace (nT²) | Trace 下降 (%) |',
             '|---|---|---|---|---|---|---|---|']
    full_metadata = json.loads((Path(full_training) / 'metadata.json').read_text())
    no_b_metadata = json.loads((Path(no_b_training) / 'metadata.json').read_text())
    notes = []
    if full_metadata['scaler'] != no_b_metadata['scaler']:
        difference = max(float(np.max(np.abs(np.asarray(full_metadata['scaler'][key]) -
                                            np.asarray(no_b_metadata['scaler'][key]))))
                         for key in ('mean', 'std', 'd_ref'))
        notes.append(f'Scaler 跨机器最大绝对差 {difference:.3e}；等价检查采用 rtol=atol=1e-10，'
                     '每组评估严格使用自身 Checkpoint 保存的 Scaler，未重新拟合或替换。')
    if full_metadata.get('gpu') != no_b_metadata.get('gpu'):
        notes.append(f"训练 GPU 不同：Full 为 {full_metadata.get('gpu')}，No-B 为 {no_b_metadata.get('gpu')}。"
                     '软件版本及数据核验见执行记录；跨硬件训练差异是本次对照的额外限制。')
    for row in summaries:
        values = ' | '.join(fmt(row, key) for key in ('initial_rmse', 'final_rmse', 'rmse_decline_pct',
                                                      'initial_trace', 'final_trace', 'trace_decline_pct'))
        lines.append(f"| {row['scenario']} | {row['method']} | {values} |")
    lines += ['', '| 场景 | 策略 | 不同网格数（含起点） | 合法重复访问 | 越界动作 | prior RMSE (nT) | prior Trace (nT²) |',
              '|---|---|---|---|---|---|---|']
    for row in summaries:
        values = ' | '.join(fmt(row, key) for key in ('unique_cells', 'repeat_visits', 'outside_actions', 'prior_rmse', 'prior_trace'))
        lines.append(f"| {row['scenario']} | {row['method']} | {values} |")
    lines += ['', '重复访问只计合法到达已访问网格；越界动作被拒绝并原地不动，两者不等价。', '',
              '| 场景 | 策略 | ' + ' | '.join(f'{i}: {tuple(delta)} (%)' for i, delta in enumerate(ACTIONS.tolist())) + ' |',
              '|---|---|' + '---|' * 8]
    for row in summaries:
        lines.append(f"| {row['scenario']} | {row['method']} | " +
                     ' | '.join(fmt(row, f'action_{i}_pct') for i in range(8)) + ' |')
    lines += ['', '## 初步结论', '']
    for scenario in ('zhurong', 'perseverance'):
        full, no_b = [next(row for row in summaries if row['scenario'] == scenario and row['method'] == method)
                      for method in ('Full', 'No-B')]
        lines.append(f"- {scenario}：No-B 相对 Full 的 RMSE 下降率差为 {no_b['rmse_decline_pct_mean'] - full['rmse_decline_pct_mean']:+.4f} 个百分点；"
                     f"Trace 下降率差为 {no_b['trace_decline_pct_mean'] - full['trace_decline_pct_mean']:+.4f} 个百分点；"
                     f"不同网格数均值差为 {no_b['unique_cells_mean'] - full['unique_cells_mean']:+.4f}。")
    identical = all(
        all(a[key] == b[key] for key in ('step', 'action', 'i', 'j'))
        for seed in range(1000, 1050)
        for a, b in zip(groups[('zhurong', 'No-B')]['paths'][seed],
                        groups[('perseverance', 'No-B')]['paths'][seed]))
    if identical:
        lines += ['', 'No-B 在两个场景的全部 50 条配对轨迹完全一致，符合位置输入策略的定义；'
                  '它学到的位置策略迁移到另一磁场仍有较高覆盖，不代表策略会根据测试场磁场或 GP 后验自适应。']
    lines += ['', '结果仅描述这一个训练种子的输入消融；不能单凭该结果确证磁场输入造成过拟合。跨场景绝对 RMSE 不直接用于判定强弱。'
              '训练期毅力评估仅用于诊断，没有据此选权重或调参；正式泛化结论需另留独立测试协议。', '',
              '## 图表', '',
              '代表轨迹预先固定 seed=1000，没有按结果挑选案例；曲线使用全部 50 种子均值±总体标准差。']
    for name in ('paired_trajectories_seed_1000', 'zhurong_step_metrics', 'perseverance_step_metrics', 'training_perseverance_rmse'):
        prefix = Path(os.path.relpath(output / 'diagnostic_figures' / name, report_path.parent)).as_posix()
        lines += ['', f'![{name}]({prefix}.png)', f'[PDF]({prefix}.pdf)']
    lines += ['', '## 执行命令', '', '环境：mars_gp_ipp。以下为本报告对应结果的复现命令；实际执行记录见验证文档。', '', '```bash',
              f'python scripts/train.py --config configs/ablation/no_b_rbf.yaml --output {no_b_training} --formats png pdf']
    for (scenario, method), group in groups.items():
        if method == 'No-B':
            lines.append(f'python scripts/evaluate.py --checkpoint {Path(no_b_training) / "checkpoint.pt"} '
                         f'--test-scenario {scenario} --output {group["directory"]} --formats png pdf')
    command = 'python scripts/no_b_diagnostics.py'
    for (scenario, method), group in groups.items():
        prefix = 'full' if method == 'Full' else 'no-b'
        command += f' --{prefix}-{scenario} {group["directory"]}'
    command += f' --full-training {full_training} --no-b-training {no_b_training} --output {output}'
    lines += [command, '```', '', '## 异常与兼容处理', '']
    lines += notes or ['未检测到配置、数据或配对协议异常。']
    lines += ['', '逐种子数据及全部动作频率见 diagnostic_per_seed.csv。']
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines) + '\n')
