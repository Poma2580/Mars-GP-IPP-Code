"""V3.5 repeat-weight sensitivity batches with frozen references and CSV reports."""
from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.envs.observations import GPTraceScaler
from mars_ipp.evaluation.evaluator import evaluate
from mars_ipp.field.field_model import FieldModel
from mars_ipp.utils.config import ROOT, load_config
from mars_ipp.utils.logger import write_csv, write_json
from mars_ipp.utils.seed import seed_all
from mars_ipp.visualization.learning_curves import read_csv
from .v3_masked import (audit_scene, audit_summary, audit_training, check_outputs,
                        plot_training, singular_figures)
from .v3_pipeline import (METRICS, empty_output, load_policy, scene_report, sha,
                          train_variant, validate_scene, verify_hashes)
from .v3_variants import scaler_for

LOG = logging.getLogger(__name__)
WEIGHTS = dict(zip(('V3_5_1', 'V3_5_2', 'V3_5_3', 'V3_5_4'), (.0275, .0412, .0550, .0687)))
LABELS = {name: name.replace('_', '.') for name in WEIGHTS}
METHODS = ('GP-Trace', 'V3.5', *LABELS.values())
BATCH_WEIGHTS = {'1-4': WEIGHTS, '5-10': dict(zip(
    (f'V3_5_{i}' for i in range(5, 11)), (.0825, .0962, .1099, .1237, .1374, .1512))),
    '11-12': {'V3_5_11': .34355, 'V3_5_12': .68710}}
BATCH_SLUGS = {'1-4': ('v_3.5.1_v3.5.4_train', 'v3.5.1_v.3.5.4_test'),
               '5-10': ('v_3.5.5_v3.5.10_train', 'v3.5.5_v.3.5.10_test'),
               '11-12': ('v_3.5.11_v3.5.12_train', 'v3.5.11_v.3.5.12_test')}
BATCH_SNAPSHOTS = {'1-4': 'logs/v3_5_rep_weight_implementation/historical_sha256.json',
                   '5-10': 'logs/v3_5_5_v3_5_10_implementation/historical_sha256.json',
                   '11-12': 'logs/v3_5_11_v3_5_12_implementation/historical_sha256.json'}


def batch_methods(batch='1-4'):
    return ('GP-Trace', 'V3.5', *(name.replace('_', '.') for name in BATCH_WEIGHTS[batch]))
FROZEN = ROOT / 'results/main/v_3.5_train/checkpoint.pt'
REFERENCE = ROOT / 'results/comparison/v_3.5_test'
NOTE = ('Only repeat penalty differs from V3.5. Training seed=0; one training run per weight. '
        'Known per-map kernel and Bmax priors. Trr=Trace_current/Trace_prior*100%; lower RMSE/Trr is better. '
        'Population SD ddof=0; 50 paired evaluation seeds/map in formal runs. '
        'Five-map averages are descriptive, not independent pooled repetitions. '
        'Higher repeat penalties need not improve RMSE/Trr; masking already prevents outside actions.')


def directories(train_root, test_root, smoke=False, batch='1-4'):
    suffix = '_smoke' if smoke else ''
    train, test = BATCH_SLUGS[batch]
    return Path(train_root) / (train + suffix), Path(test_root) / (test + suffix)


def configs(batch='1-4'):
    base = load_config(ROOT / 'configs/main/v3_5.yaml')
    result = {}
    for name, weight in BATCH_WEIGHTS[batch].items():
        c = load_config(ROOT / f'configs/main/{name}.yaml')
        expected = deepcopy(base)
        expected['experiment'] = name
        expected['reward']['weights'] = [1., weight, .02]
        if c != expected:
            raise ValueError(f'Only experiment and repeat weight may differ: {name}')
        result[name] = c
    return base, result


def network_sha(network):
    digest = hashlib.sha256()
    for key, value in sorted(network.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def preflight(train_root, test_root, stage='preflight', smoke=False, batch='1-4'):
    base, variants = configs(batch)
    if not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA device is unavailable')
    _, payload = load_policy(FROZEN)
    saved = yaml.safe_load((FROZEN.parent / 'others/config.yaml').read_text())
    if payload['config'] != base or saved != base or payload['progress'] != dict(episode=10000, steps=700000, completed=True):
        raise ValueError('Frozen V3.5 configuration/progress mismatch')
    acceptance = json.loads((FROZEN.parent / 'others/acceptance.json').read_text())
    if not acceptance['passed'] or acceptance['checkpoint_sha256'] != sha(FROZEN):
        raise ValueError('Frozen V3.5 acceptance/hash mismatch')
    fields = {}
    for kind in ('train', 'test'):
        for i in range(1, 6):
            name = f'{kind}{i}'; path = ROOT / f'data/models/{name}.pkl'
            field = FieldModel.load(path)
            if not np.isfinite(field.truth(np.array([[0., 0.], [30., 30.]]))).all():
                raise ValueError(f'Nonfinite field: {name}')
            fields[name] = {'path': str(path), 'sha256': sha(path), 'parameters': field.parameters}
    train, test = directories(train_root, test_root, smoke, batch)
    check_outputs([*([] if stage == 'evaluate' else [train]), *([] if stage == 'train' else [test])])
    report = dict(batch=batch, configs=variants, fields=fields, cuda=torch.cuda.get_device_name(), smoke=smoke,
                  frozen={'path': str(FROZEN), 'sha256': sha(FROZEN)}, note=NOTE)
    report['reference_csv'] = {}
    for i in range(1, 6):
        for method in METHODS[:2]:
            key = f'test{i}/{method}'
            try:
                _, _, provenance = cached_reference(f'test{i}', method, list(range(1000, 1050)), report)
                report['reference_csv'][key] = provenance
            except (OSError, KeyError, ValueError, AssertionError) as exc:
                report['reference_csv'][key] = {'reused': False, 'reevaluate_reason': str(exc)}
    return report


def typed_rows(path):
    """Restore scalar CSV types required by trajectory/reward audits."""
    rows = read_csv(path)
    for row in rows:
        for key, value in row.items():
            if value in ('True', 'False'):
                row[key] = value == 'True'
            else:
                try:
                    row[key] = float(value)
                except (ValueError, TypeError):
                    pass
    return rows


def audit_method(method, rows, paths, seeds, weight=None):
    # Reuse the existing action replay with V3.5 semantics for each new policy.
    label = 'V3.5' if method != 'GP-Trace' else method
    audit_scene({label: rows}, {label: paths}, seeds)
    if {int(r['seed']) for r in rows} != set(seeds) or len(rows) != len(seeds) * 72 or len(paths) != len(seeds) * 70:
        raise ValueError('Unexpected seeds or record count')
    if method != 'GP-Trace':
        for row in rows:
            if row['stage'] != 'step':
                continue
            if row['outside_actions'] != 0 or row['r_rep'] not in (0., -1.):
                raise ValueError('Invalid repeat/mask record')
            np.testing.assert_allclose([row[f'weighted_r_{k}'] for k in ('gp', 'rep', 'mh')],
                                      np.array([row[f'r_{k}'] for k in ('gp', 'rep', 'mh')]) * [1., weight, .02], atol=1e-12)
            np.testing.assert_allclose(row['reward_total'], sum(row[f'weighted_r_{k}'] for k in ('gp', 'rep', 'mh')), atol=1e-10)
            if row['r_rep'] != -float(not row['sampled']):
                raise ValueError('Repeat indicator mismatch')


def cached_reference(scene, method, seeds, report):
    metadata = json.loads((REFERENCE / 'others/metadata.json').read_text())
    if metadata['smoke'] or metadata['seeds'] != list(range(1000, 1050)):
        raise ValueError('Reference seeds/protocol mismatch')
    if metadata['data'][scene]['sha256'] != report['fields'][scene]['sha256']:
        raise ValueError('Reference field hash mismatch')
    if metadata['checkpoints']['V3.5']['sha256'] != report['frozen']['sha256']:
        raise ValueError('Reference policy hash mismatch')
    if metadata['configs']['v3_5'] != load_config(ROOT / 'configs/main/v3_5.yaml'):
        raise ValueError('Reference configuration mismatch')
    if metadata['configs']['v3_3_4'] != load_config(ROOT / 'configs/main/v3_3_4.yaml'):
        raise ValueError('Reference GP-Trace configuration mismatch')
    # Original execution manifest contains hashes of every protected historical CSV.
    manifest = json.loads((ROOT / BATCH_SNAPSHOTS[report.get('batch', '1-4')]).read_text())
    source = REFERENCE / scene / 'others' / method
    files = [source / name for name in ('eval_metrics.csv', 'trajectories.csv')]
    hashes = {str(p): sha(p) for p in files}
    for p in files:
        if manifest.get(str(p.resolve())) != hashes[str(p)]:
            raise ValueError('Reference CSV hash mismatch')
    rows, paths = [typed_rows(p) for p in files]
    rows = [r for r in rows if int(r['seed']) in seeds]
    paths = [r for r in paths if int(r['seed']) in seeds]
    audit_method(method, rows, paths, seeds, .014)
    return rows, paths, {'reused': True, 'files': hashes}


def report_scene(scene, runs, paths, target, seeds, smoke, methods=METHODS):
    validate_scene(runs, paths, seeds, methods)
    summary = scene_report(scene, runs, paths, target, seeds, smoke, methods=methods,
                           note=NOTE, weight_report=True, sample_label=f'{len(seeds)} paired seeds/map',
                           trajectory_layout=(2, 4) if len(methods) == 8 else None)
    singular_figures(target)
    audit_summary(target, runs)
    return summary


def summary_report(test, smoke, methods=METHODS, train=None):
    summary = []
    for i in range(1, 6):
        scene = f'test{i}'
        rows = typed_rows(test / scene / 'others/summary.csv')
        summary.extend(dict(map=scene, **row) for row in rows if int(row['step']) == 70)
    if len(summary) != 5 * len(methods):
        raise ValueError('Incomplete five-map summary')
    averages = []
    for method in methods:
        selected = [r for r in summary if r['method'] == method]
        averages.append(dict(map='five_map_mean', method=method, step=70, runs=selected[0]['runs'],
                             **{k: float(np.mean([r[k] for r in selected])) for k in selected[0] if k.endswith(('_mean', '_std'))}))
    write_csv(test / 'others/summary_70.csv', summary + averages)
    weights = {'V3.5': .014, **{n.replace('_', '.'): w for group in BATCH_WEIGHTS.values() for n, w in group.items()}}
    ranking = [dict(rank=index, repeat_weight=weights.get(row['method']), **row)
               for index, row in enumerate(sorted(averages, key=lambda row: row['trr_mean']), 1)]
    write_csv(test / 'others/trr_ranking.csv', ranking)
    best_new = min(averages[2:], key=lambda row: row['trr_mean'])
    write_json(test / 'others/selection.json', dict(primary_metric='five-map mean Trr at step 70',
               training_seed=0, checkpoint='final', independent_validation=False, smoke=smoke,
               best_new=best_new, best_all=ranking[0]))
    lines = ['# 重复惩罚权重比较' + ('（冒烟，非正式结果）' if smoke else ''), '', NOTE, '',
             f'每图每方法 50 次正式评估；五地图每方法 250 次，{len(methods)} 方法共 {250*len(methods)} 条轨迹。' if not smoke else '每图每方法 2 次冒烟评估。',
             '五地图平均为各地图均值和总体标准差的分别等权平均，不是合并样本的标准差。', '',
             '| 地图 | 方法 | RMSE (nT) | Trr (%) | 重复 | 越界 |', '|---|---|---|---|---|---|']
    for r in summary + averages:
        cells = [f"{r[k+'_mean']:.4f} ± {r[k+'_std']:.4f}" for k in ('rmse', 'trr', 'repeated_visits', 'outside_actions')]
        lines.append(f"| {r['map']} | {r['method']} | " + ' | '.join(cells) + ' |')
    lines += ['', '## 结果观察', '']
    base = next(r for r in averages if r['method'] == 'V3.5')
    for r in averages[2:]:
        lines.append(f"- {r['method']} 相对 V3.5：五地图平均 RMSE 差 {r['rmse_mean']-base['rmse_mean']:+.4f} nT，"
                     f"Trr 差 {r['trr_mean']-base['trr_mean']:+.4f} 个百分点，重复次数差 {r['repeated_visits_mean']-base['repeated_visits_mean']:+.4f}。")
    lines += ['', '以上为单次训练 seed=0 下的描述性结果；需结合各地图和训练奖励分量判断，不能推断权重越大越优。',
              'Mask 已使 V3.5 越界为零，不将新模型的零越界解释为进一步改善。']
    lines += ['', '## 按五地图70步平均 Trr 排名', '',
              '| 排名 | 方法 | 重复惩罚系数 | Trr (%) | RMSE (nT) |', '|---|---|---|---|---|']
    for row in ranking:
        coefficient = row['repeat_weight']
        precision = 5 if coefficient is not None and round(coefficient, 4) != coefficient else 4
        weight = '—' if coefficient is None else f"{coefficient:.{precision}f}"
        lines.append(f"| {row['rank']} | {row['method']} | {weight} | {row['trr_mean']:.4f} | {row['rmse_mean']:.4f} |")
    lines += ['', f"本轮新版本中 Trr 最低：{best_new['method']}；Trr={best_new['trr_mean']:.4f}%，"
              f"RMSE={best_new['rmse_mean']:.4f} nT。", '',
              '本轮在现有五张测试地图上筛选权重，不将结果视为独立验证的泛化最优。']
    if train is not None:
        diagnostics = []
        for method in methods[2:]:
            rows = typed_rows(Path(train) / method.lower() / 'others/train_metrics.csv')[-1000:]
            keys = ('reward', 'weighted_r_gp', 'weighted_r_rep', 'weighted_r_mh', 'repeated_visits')
            means = {key: float(np.mean([r[key] for r in rows])) for key in keys}
            diagnostics.append(dict(method=method, episodes=len(rows), repeat_weight=weights[method],
                negative_reward_fraction=float(np.mean([r['reward'] < 0 for r in rows])),
                repeat_penalty_to_gp=abs(means['weighted_r_rep']) / max(abs(means['weighted_r_gp']), 1e-15), **means))
        write_csv(test / 'others/training_reward_diagnostics.csv', diagnostics)
        lines += ['', '## 训练末期奖励诊断', '', '正式训练取最后1000回合；冒烟取全部回合。比值为平均重复惩罚绝对值/平均GP奖励。', '',
                  '| 方法 | 平均总奖励 | GP | 重复惩罚 | MH | 惩罚/GP | 负总奖励回合比例 |', '|---|---|---|---|---|---|---|']
        for row in diagnostics:
            cells = [f"{row[key]:.4f}" for key in ('reward', 'weighted_r_gp', 'weighted_r_rep', 'weighted_r_mh', 'repeat_penalty_to_gp')]
            lines.append(f"| {row['method']} | " + ' | '.join(cells) + f" | {row['negative_reward_fraction']:.2%} |")
    (test / 'others/weight_comparison.md').write_text('\n'.join(lines) + '\n')


def audit_variant(directory, name, smoke, config):
    result = audit_training(directory, 'v3_5', smoke, expected_config=config)
    agent, _ = load_policy(directory / 'checkpoint.pt')
    metadata = json.loads((directory / 'others/metadata.json').read_text())
    if network_sha(agent.network) == metadata['initial_network_sha256']:
        raise ValueError('Network did not change during training')
    rows = typed_rows(directory / 'others/train_metrics.csv')
    if [r['map'] for r in rows] != [f'train{i % 5 + 1}' for i in range(len(rows))]:
        raise ValueError('Incorrect training map rotation')
    for row in rows:
        np.testing.assert_allclose(row['trr'], 100 * row['trace_current'] / row['trace_prior'], atol=1e-10)
    updates = typed_rows(directory / 'others/updates.csv')
    if [int(r['samples']) for r in updates] != ([2000, 100] if smoke else [2000] * 350):
        raise ValueError('Incorrect rollout update sizes')
    result.update(initial_network_sha256=metadata['initial_network_sha256'],
                  final_network_sha256=network_sha(agent.network), experiment=name)
    write_json(directory / 'others/summary.json', result)
    write_json(directory / 'others/acceptance.json', result)
    return result


def train_all(train, variants, smoke):
    empty_output(train)
    results = {}
    for name, config in variants.items():
        directory = train / name.replace('_', '.').lower()
        LOG.info('Training %s smoke=%s', name, smoke)
        seed_all(config['seed'])
        initial = PPO(config['ppo'], scaler_for(config))
        initial_hash = network_sha(initial.network)
        del initial
        train_variant(config, directory, smoke, plotter=plot_training)
        metadata_path = directory / 'others/metadata.json'
        metadata = json.loads(metadata_path.read_text())
        metadata.update(note=NOTE, experiment=name, initialization='fresh PPO; seed=0; no checkpoint restored',
                        initial_network_sha256=initial_hash)
        write_json(metadata_path, metadata)
        results[name] = audit_variant(directory, name, smoke, config)
        write_json(directory / 'others/acceptance.json', results[name])
    if len({r['final_network_sha256'] for r in results.values()}) != len(variants):
        raise ValueError('Variants share identical final network weights')
    write_json(train / 'others/acceptance.json', dict(passed=True, smoke=smoke, variants=results))


def evaluate_all(train, test, report, smoke):
    methods = batch_methods(report.get('batch', '1-4'))
    agents = {}
    for name, config in report['configs'].items():
        directory = train / name.replace('_', '.').lower()
        audit_variant(directory, name, smoke, config)
        agents[name.replace('_', '.')] = load_policy(directory / 'checkpoint.pt')
    seeds = [1000, 1001] if smoke else list(range(1000, 1050))
    empty_output(test)
    frozen_agent, frozen_payload = load_policy(FROZEN)
    provenance = {}
    for i in range(1, 6):
        scene = f'test{i}'; field = FieldModel.load(ROOT / f'data/models/{scene}.pkl')
        runs = {}; paths = {}; provenance[scene] = {}
        for method in methods:
            if method in ('GP-Trace', 'V3.5'):
                try:
                    rows, trajectory, source = cached_reference(scene, method, seeds, report)
                except (OSError, KeyError, ValueError, AssertionError) as exc:
                    LOG.warning('Reevaluating frozen %s/%s: %s', scene, method, exc)
                    cfg = deepcopy(frozen_payload['config'])
                    cfg['field']['test_scenario'] = scene; cfg['evaluation']['seeds'] = seeds
                    if method == 'GP-Trace':
                        cfg = load_config(ROOT / 'configs/main/v3_3_4.yaml')
                        cfg['field']['test_scenario'] = scene; cfg['evaluation']['seeds'] = seeds
                        rows, trajectory, _ = evaluate(field, GPTraceScaler(), cfg, algorithm='gp_trace')
                    else:
                        rows, trajectory, _ = evaluate(field, frozen_agent.scaler, cfg, frozen_agent, episode=10000)
                    source = {'reused': False, 'reason': str(exc)}
                weight = .014
            else:
                agent, payload = agents[method]
                cfg = deepcopy(payload['config'])
                cfg['field']['test_scenario'] = scene; cfg['evaluation']['seeds'] = seeds
                rows, trajectory, _ = evaluate(field, agent.scaler, cfg, agent, episode=payload['progress']['episode'])
                weight = cfg['reward']['weights'][1]
                checkpoint = train / method.lower() / 'checkpoint.pt'
                source = {'checkpoint': str(checkpoint), 'sha256': sha(checkpoint)}
            audit_method(method, rows, trajectory, seeds, weight)
            target = test / scene / 'others' / method
            write_csv(target / 'eval_metrics.csv', rows); write_csv(target / 'trajectories.csv', trajectory)
            provenance[scene][method] = source
            runs[method] = typed_rows(target / 'eval_metrics.csv')
            paths[method] = typed_rows(target / 'trajectories.csv')
        report_scene(scene, runs, paths, test / scene, seeds, smoke, methods)
        LOG.info('Completed %d-method paired evaluation %s', len(methods), scene)
    summary_report(test, smoke, methods, train)
    write_json(test / 'others/metadata.json', dict(**report, seeds=seeds, methods=methods, provenance=provenance))
    write_json(test / 'others/acceptance.json', dict(passed=True, smoke=smoke, maps=5, methods=len(methods),
               seeds_per_map=len(seeds), steps=70, trajectories=5*len(methods)*len(seeds), new_ppo_outside_actions=0))


def replot(train, test, output, smoke, batch='1-4'):
    check_outputs([output]); output = Path(output)
    methods = batch_methods(batch)
    for label in methods[2:]:
        plot_training(train / label.lower(), label, output / 'main' / label.lower())
    seeds = [1000, 1001] if smoke else list(range(1000, 1050))
    for i in range(1, 6):
        scene = f'test{i}'
        runs = {m: typed_rows(test / scene / f'others/{m}/eval_metrics.csv') for m in methods}
        paths = {m: typed_rows(test / scene / f'others/{m}/trajectories.csv') for m in methods}
        report_scene(scene, runs, paths, output / 'comparison' / scene, seeds, smoke, methods)
    summary_report(output / 'comparison', smoke, methods, train)


def history_manifest(exclude):
    files = [p for folder in ('results', 'configs', 'data/models') for p in (ROOT / folder).rglob('*')
             if p.is_file() and not any(e.resolve() in p.resolve().parents for e in exclude)]
    return {str(p): sha(p) for p in files}


def run(stage, train_root, test_root, output=None, smoke_results=False, batch='1-4'):
    smoke = stage == 'smoke' or smoke_results
    train, test = directories(train_root, test_root, smoke, batch)
    if stage == 'plot':
        return replot(train, test, output, smoke, batch)
    report = preflight(train_root, test_root, stage, smoke, batch)
    if stage == 'preflight':
        print(json.dumps(report, indent=2)); return report
    protected = history_manifest([train, test])
    started = time.monotonic()
    try:
        if stage in ('train', 'smoke'):
            train_all(train, report['configs'], smoke)
        if stage in ('evaluate', 'smoke'):
            evaluate_all(train, test, report, smoke)
        verify_hashes(protected)
        original = ROOT / BATCH_SNAPSHOTS[batch]
        if original.exists():
            verify_hashes(json.loads(original.read_text()))
        target = train if stage == 'train' else test
        write_json(target / 'others/execution.json', dict(passed=True, stage=stage, seconds=time.monotonic()-started,
                   history_verified=True, protected_sha256=protected))
    except Exception as exc:
        for directory in (train, test):
            if directory.exists():
                write_json(directory / 'others/pipeline_failure.json', dict(stage=stage, error=repr(exc)))
        raise
