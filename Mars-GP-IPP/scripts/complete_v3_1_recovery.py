"""Complete and verify the reconstructed V3.1 seed-0 comparisons."""
import _bootstrap
import csv
import json
from pathlib import Path
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from evaluate import run as evaluate_run
from plot_trajectories import run as plot_trajectories
from run_test_maps_comparison import DEFAULT_DATA, MAPS, validate_rows
from run_v3_gp_trace import (checkpoint_for, seal_artifacts, sha256,
                             validate_artifacts, validate_evaluation,
                             validate_paired_starts)
from run_v3_1_global_raw import validate_raw_evaluation
from mars_ipp.field.field_model import FieldModel
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.utils.config import ROOT, load_config, merge
from mars_ipp.utils.logger import write_csv, write_json
from mars_ipp.visualization.comparison import plot_comparison, _step_statistics
from mars_ipp.visualization.learning_curves import read_csv, save_figure

NAME = 'v3_1_gp_trace_global_raw_train1_recovery'
OUTPUT = ROOT / 'results/comparison' / NAME
TRAINING = ROOT / 'results/main' / NAME / 'seed_0'
SOURCE = ROOT / 'results/comparison/v3_gp_trace_train1'
CHECKPOINT = TRAINING / 'checkpoint.pt'
V3_CHECKPOINT = ROOT / 'results/main/v3_gp_trace_train1/seed_0/checkpoint.pt'
SEEDS = tuple(range(1000, 1050))
METHODS = ('gp_trace', 'v3', 'ppo_seed_0')
METRICS = ('rmse', 'trr', 'trace_current', 'unique_positions',
           'repeated_visits', 'outside_actions')
LABELS = {'gp_trace': 'GP-Trace', 'v3': 'V3 local raw', 'ppo_seed_0': 'V3.1 global raw'}


def input_hashes():
    paths = [CHECKPOINT, V3_CHECKPOINT, ROOT / 'data/models/train1.pkl',
             Path(load_config(ROOT / 'configs/main/v3_1_gp_trace_global_raw_train1.yaml')['v3']['original_source'])]
    paths.extend(DEFAULT_DATA / f'{scene}.pkl' for scene in MAPS)
    paths.extend(path for method in ('gp_trace', 'ppo_seed_0')
                 for path in (SOURCE / method).rglob('*') if path.is_file())
    paths.extend(path for scene in MAPS for path in
                 (ROOT / 'results/comparison/test_maps_seed0' / scene / 'gp_trace').rglob('*') if path.is_file())
    return {str(path): sha256(path) for path in paths}


def copy_verified(source, destination):
    if destination.exists():
        if {p.name: sha256(p) for p in destination.iterdir() if p.is_file()} != {
                p.name: sha256(p) for p in source.iterdir() if p.is_file()}:
            raise ValueError(f'Existing reused output differs: {destination}')
        return
    shutil.copytree(source, destination)


def validate_saved_map(directory, method, field, config):
    rows = read_csv(directory / 'eval_metrics.csv')
    paths = read_csv(directory / 'trajectories.csv')
    expected_algorithm = 'gp_trace' if method == 'gp_trace' else 'ppo'
    expected_episode = 0 if method == 'gp_trace' else 10000
    if yaml.safe_load((directory / 'config.yaml').read_text()) != config:
        raise ValueError(f'Map evaluation config mismatch: {directory}')
    for row in (*rows, *paths):
        if row['algorithm'] != expected_algorithm:
            raise ValueError('Unexpected evaluator algorithm')
        row['algorithm'] = method
    validate_rows(rows, paths, SEEDS, method, expected_episode, field)
    if method == 'ppo_seed_0':
        for seed in SEEDS:
            selected = [r for r in rows if int(r['seed']) == seed]
            reward = sum(float(r['r_gp']) for r in selected[2:])
            delta = float(selected[1]['trace_current']) - float(selected[-1]['trace_current'])
            if not np.isclose(reward, delta, atol=1e-5, rtol=2e-7):
                raise ValueError(f'V3.1 map Trace identity fails: {directory}, seed={seed}')
    return rows, paths


def endpoint_tables(scene, directory, rows_by_method):
    summaries, differences = [], []
    endpoints = {}
    starts = {}
    for method in METHODS:
        rows = rows_by_method[method]
        starts[method] = {int(r['seed']): (int(r['start_i']), int(r['start_j']),
                                         float(r['trace_current']))
                          for r in rows if r['stage'] == 'reset'}
        endpoints[method] = {int(r['seed']): r for r in rows
                             if r['stage'] == 'step' and int(r['step']) == 70}
        if set(endpoints[method]) != set(SEEDS):
            raise ValueError(f'Incomplete 50-seed endpoints: {scene}/{method}')
        summary = {'scenario': scene, 'method': method, 'training_seed': '' if method == 'gp_trace' else 0,
                   'test_runs': 50}
        for metric in METRICS:
            values = np.array([float(endpoints[method][seed][metric]) for seed in SEEDS])
            summary[f'{metric}_mean'] = float(values.mean())
            summary[f'{metric}_std'] = float(values.std(ddof=0))
            steps, means, stds = _step_statistics(rows, rows[0]['algorithm'], metric)
            if steps != list(range(71)) or not np.allclose(
                    [means[-1], stds[-1]], [summary[f'{metric}_mean'], summary[f'{metric}_std']],
                    atol=1e-12, rtol=1e-12):
                raise ValueError(f'Plot and endpoint statistics disagree: {scene}/{method}/{metric}')
        summaries.append(summary)
    reference = starts['gp_trace']
    for method in METHODS[1:]:
        for seed in SEEDS:
            a, b = reference[seed], starts[method][seed]
            if a[:2] != b[:2] or not np.isclose(a[2], b[2], atol=1e-8, rtol=1e-10):
                raise ValueError(f'Unpaired start or GP posterior: {scene}/{method}/{seed}')
    for seed in SEEDS:
        for comparator in ('gp_trace', 'v3'):
            row = {'scenario': scene, 'seed': seed, 'comparison': 'v3_1_minus_' + comparator}
            row.update({metric + '_difference': float(endpoints['ppo_seed_0'][seed][metric])
                        - float(endpoints[comparator][seed][metric]) for metric in METRICS})
            differences.append(row)
    summary_dir = directory / 'summary'
    write_csv(summary_dir / 'comparison.csv', summaries)
    write_csv(summary_dir / 'paired_differences.csv', differences)
    write_json(summary_dir / 'validation.json', {'scenario': scene, 'seeds': list(SEEDS),
               'paired_starts': True, 'std_ddof': 0, 'reconstructed_checkpoint_sha256': sha256(CHECKPOINT),
               'methods': list(METHODS)})
    runs = {method: {'rows': rows_by_method[method], 'algorithm': rows_by_method[method][0]['algorithm'],
                     'config': yaml.safe_load((directory / method / 'config.yaml').read_text())}
            for method in METHODS}
    plot_comparison(runs, directory / 'figures', ('png', 'pdf'), methods=METHODS)
    with (summary_dir / 'comparison.tex').open('w') as stream:
        stream.write('\\begin{tabular}{lrrrrrr}\nMethod & RMSE & Trr & Trace & Unique & Repeat & Outside \\\\\n\\hline\n')
        for row in summaries:
            values = ' & '.join(f"${row[f'{metric}_mean']:.4f} \\pm {row[f'{metric}_std']:.4f}$"
                                for metric in METRICS)
            stream.write(f"{LABELS[row['method']]} & {values} \\\\\n")
        stream.write('\\end{tabular}\n')
    return summaries


def plot_map_trajectories(scene, directory):
    for seed in (1000, 1001, 1002):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, method in zip(axes, METHODS):
            metric_rows = read_csv(directory / method / 'eval_metrics.csv')
            reset = next(r for r in metric_rows if int(r['seed']) == seed and r['stage'] == 'reset')
            path_rows = [r for r in read_csv(directory / method / 'trajectories.csv')
                         if int(r['seed']) == seed]
            xy = np.array([(int(reset['start_i']), int(reset['start_j'])),
                           *((int(r['i']), int(r['j'])) for r in path_rows)]) * 2
            ax.plot(xy[:, 0], xy[:, 1], '-o', linewidth=1, markersize=2)
            ax.scatter(*xy[0], c='green', s=65, zorder=5, label='Start')
            ax.scatter(*xy[-1], c='gold', s=65, zorder=5, label='End')
            positions, counts = np.unique(xy, axis=0, return_counts=True)
            for position, count in zip(positions, counts):
                if count > 1:
                    ax.annotate(str(count), position, fontsize=7, color='blue')
            ax.set(xlim=(-1, 31), ylim=(-1, 31), xlabel='Local x', ylabel='Local y',
                   title=f'{LABELS[method]}: {len(positions)} unique')
            ax.set_aspect('equal')
            ax.grid(alpha=.3)
        axes[-1].legend(fontsize=8)
        fig.suptitle(f'{scene}: paired seed {seed}, 70 actions')
        save_figure(fig, directory / 'figures' / f'trajectories_seed_{seed}', ('png', 'pdf'))


def historical_check(summaries):
    old_log = ROOT / 'logs/v3_test2_test5.log'
    content = old_log.read_text()
    old_hashes = set(re.findall(r'v3_1_gp_trace_global_raw_train1/seed_0/checkpoint\.pt; sha256=([0-9a-f]{64})', content))
    if old_hashes != {sha256(CHECKPOINT)}:
        raise ValueError('Reconstructed checkpoint differs from the original V3.1 log')
    checked = {}
    for scene in MAPS:
        matches = re.findall(rf'{scene} endpoint_mean_std_ddof0=(\{{[^\n]+\}})', content)
        if not matches:
            raise ValueError(f'Original V3.1 map statistics unavailable: {scene}')
        original = json.loads(matches[-1])['v3_1']
        current = next(row for row in summaries if row['scenario'] == scene and row['method'] == 'ppo_seed_0')
        for metric in ('rmse', 'trr'):
            np.testing.assert_allclose(original[metric],
                [current[f'{metric}_mean'], current[f'{metric}_std']], atol=1e-12, rtol=1e-12)
        checked[scene] = True
    return {'original_checkpoint_sha256_matches': True,
            'original_map_rmse_trr_statistics_match': checked,
            'source_log': str(old_log.relative_to(ROOT))}


def write_report(summaries, historical):
    lines = ['# V3.1 恢复实验：50-seed 正式测试', '',
             'V3.1 seed 0 使用原配置重训 10000 回合；这是重新生成的产物，并非找回已删除文件。',
             f'重训 checkpoint SHA256 `{sha256(CHECKPOINT)}` 与删除前日志记录的原 checkpoint 完全相同。',
             'train1 和 test2–test5 各用 seeds 1000–1049、70 步，三方法逐 seed 起点及重置后 GP Trace 已校验一致。',
             'V3.1 用全局原始 Trace 下降奖励、γ=1；本次 V3 对照用当前局部原始 Trace 下降奖励、γ=0.99。',
             'GP-Trace 为合法八邻域局部贪心；所有表格标准差 ddof=0，RMSE 与 Trr 越低越好。', '',
             '| 场景 | 方法 | RMSE (nT) | Trr (%) | 最终 Trace (nT²) | 唯一点 | 重复 | 越界 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        cells = [f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}" for metric in METRICS]
        lines.append('| ' + ' | '.join([row['scenario'], LABELS[row['method']],
                     cells[0], cells[1], cells[2], cells[3], cells[4], cells[5]]) + ' |')
    lines += ['', 'V3.1 减当前 V3 的均值差：']
    for scene in ('train1', *MAPS):
        directory = OUTPUT if scene == 'train1' else OUTPUT / scene
        rows = [r for r in read_csv(directory / 'summary/paired_differences.csv')
                if r['comparison'] == 'v3_1_minus_v3']
        rmse = np.mean([float(r['rmse_difference']) for r in rows])
        trr = np.mean([float(r['trr_difference']) for r in rows])
        lines.append(f'- {scene}: RMSE {rmse:+.4f} nT；Trr {trr:+.4f} 个百分点。')
    lines += ['', 'train1 的 V3.1 RMSE/Trr 与先前报告的四位小数一致；'
              'test2–test5 的 V3.1 RMSE/Trr 均值和标准差与删除前日志逐项一致。',
              '当前 V3 模型不是旧的相对奖励 V3，不能将两轮 V3 对照数值直接视为同一模型。',
              '', '逐 seed 差值、完整逐步指标、轨迹与场快照位于各场景目录。'
              '每个 `summary/` 有 CSV、LaTeX 和校验记录；`figures/` 有曲线与 seeds 1000–1002 轨迹。',
              f'原始训练数据 SHA256 `{load_config(ROOT / "configs/main/v3_1_gp_trace_global_raw_train1.yaml")["v3"]["sha256"]}`。',
              f'旧日志核对：`{historical["source_log"]}`。', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines))


def run():
    cfg = load_config(ROOT / 'configs/main/v3_1_gp_trace_global_raw_train1.yaml')
    checkpoint_for(cfg, TRAINING)
    if sha256(cfg['v3']['original_source']) != cfg['v3']['sha256'] or sha256(ROOT / 'data/models/train1.pkl') != cfg['v3']['sha256']:
        raise ValueError('train1 source hash changed')
    baseline_cfg = load_config(ROOT / 'configs/main/v3_gp_trace_train1.yaml')
    inputs = input_hashes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print('train1: validating and combining three methods', flush=True)
    validate_raw_evaluation(OUTPUT / 'ppo_seed_0', cfg, CHECKPOINT)
    for source_name, method, algorithm, checkpoint in (
            ('gp_trace', 'gp_trace', 'gp_trace', None),
            ('ppo_seed_0', 'v3', 'ppo', V3_CHECKPOINT)):
        source = SOURCE / source_name
        validate_evaluation(source, baseline_cfg, checkpoint, algorithm)
        copy_verified(source, OUTPUT / method)
    validate_paired_starts(OUTPUT, SEEDS, METHODS)
    train_rows = {method: read_csv(OUTPUT / method / 'eval_metrics.csv') for method in METHODS}
    all_summaries = endpoint_tables('train1', OUTPUT, train_rows)
    for seed in (1000, 1001, 1002):
        plot_trajectories(OUTPUT, seed, methods=METHODS)
    for scene in MAPS:
        print(f'{scene}: evaluating frozen V3 and V3.1 policies', flush=True)
        field = FieldModel.load(DEFAULT_DATA / f'{scene}.pkl')
        directory = OUTPUT / scene
        directory.mkdir(exist_ok=True)
        configs = {}
        rows_by_method = {}
        for method, checkpoint, source_cfg in (
                ('v3', V3_CHECKPOINT, baseline_cfg), ('ppo_seed_0', CHECKPOINT, cfg)):
            own = merge(source_cfg, {'field': {'test_scenario': scene,
                                               'models': {scene: str(DEFAULT_DATA / f'{scene}.pkl')}},
                                     'evaluation': {'seeds': list(SEEDS)}})
            configs[method] = own
            destination = directory / method
            if not destination.exists():
                evaluate_run(own, destination, checkpoint, 'ppo', plot=False,
                             preserve_checkpoint_scaler=True)
                seal_artifacts(destination)
            validate_artifacts(destination)
            rows_by_method[method], _ = validate_saved_map(destination, method, field, own)
        source = ROOT / 'results/comparison/test_maps_seed0' / scene / 'gp_trace'
        source_rows = read_csv(source / 'eval_metrics.csv')
        source_paths = read_csv(source / 'trajectories.csv')
        validate_rows(source_rows, source_paths, SEEDS, 'gp_trace', 0, field)
        copy_verified(source, directory / 'gp_trace')
        rows_by_method['gp_trace'], _ = validate_saved_map(
            directory / 'gp_trace', 'gp_trace', field,
            yaml.safe_load((source / 'config.yaml').read_text()))
        all_summaries.extend(endpoint_tables(scene, directory, rows_by_method))
        plot_map_trajectories(scene, directory)
        print(f'{scene}: 50 paired seeds and outputs verified', flush=True)
    if any(sha256(path) != digest for path, digest in inputs.items()):
        raise ValueError('An input artifact changed during comparison')
    write_csv(OUTPUT / 'all_scenarios_comparison.csv', all_summaries)
    historical = historical_check(all_summaries)
    write_json(OUTPUT / 'recovery_manifest.json', {'status': 'complete',
               'provenance': 'deterministic retraining, not restoration of deleted files',
               'checkpoint_sha256': sha256(CHECKPOINT), 'input_sha256': inputs,
               'historical_comparison': historical, 'scenarios': ['train1', *MAPS],
               'test_seeds': list(SEEDS), 'std_ddof': 0})
    write_report(all_summaries, historical)
    print('ALL COMPARISONS COMPLETE', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
