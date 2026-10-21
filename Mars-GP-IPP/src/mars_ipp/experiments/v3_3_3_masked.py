"""Single-factor V3.3.3 mask experiment with paired, audited evaluation."""
from copy import deepcopy
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch

from mars_ipp.evaluation.evaluator import evaluate
from mars_ipp.field.field_model import FieldModel
from mars_ipp.utils.config import ROOT, load_config
from mars_ipp.utils.logger import write_csv, write_json
from mars_ipp.visualization.learning_curves import read_csv
from .v3_pipeline import (empty_output, load_policy, protect_history,
                          scene_report, sha, train_variant, verify_hashes)
from .v3_masked import audit_scene, audit_summary, audit_training, check_outputs, singular_figures
from .v3_weights import plot_training as legacy_plot_training
from .v3_rep_weights import typed_rows

LOG = logging.getLogger(__name__)
METHODS = ('V3.3.3', 'V3.3.3_m')
BASE = ROOT / 'results/main/v3_3_weight_train/V3_3_3/checkpoint.pt'
REFERENCE = ROOT / 'results/comparison/v3_3_weight_train'
NOTE = ('Only outside-action masking differs from V3.3.3; repeat weight=0.005, 11D state, no MH. '
        'Training seed=0, final checkpoint. Known per-map field/kernel priors. '
        'Trr=Trace_current/Trace_prior*100%; lower RMSE/Trr/repeats is better. '
        'Population SD ddof=0. Five-map mean and SD are separate equal-map averages, '
        'not pooled SD. Single training seed: descriptive comparison only.')


def configs():
    base = load_config(ROOT / 'configs/main/V3_3_3.yaml')
    new = load_config(ROOT / 'configs/main/V3_3_3_m.yaml')
    expected = deepcopy(base)
    expected['experiment'] = 'V3_3_3_m'
    expected['ppo']['action_mask'] = True
    if new != expected:
        raise ValueError('Only experiment name and action_mask may differ from V3.3.3')
    return base, new


def directories(train_root, test_root, smoke):
    suffix = '_smoke' if smoke else ''
    return (Path(train_root) / ('v_3.3.3_m_train' + suffix),
            Path(test_root) / ('v_3.3.3_m_test' + suffix))


def preflight(train, test, stage, smoke):
    base, new = configs()
    agent, payload = load_policy(BASE)
    if (payload['config'] != base or agent.action_mask_enabled or
            payload['progress'] != dict(episode=10000, steps=700000, completed=True)):
        raise ValueError('Invalid frozen V3.3.3 baseline')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    data = {}
    for kind in ('train', 'test'):
        for i in range(1, 6):
            name = f'{kind}{i}'; path = ROOT / f'data/models/{name}.pkl'
            field = FieldModel.load(path)
            if not np.isfinite(field.truth(np.array([[0., 0.], [30., 30.]]))).all():
                raise ValueError(f'Nonfinite field {name}')
            data[name] = dict(path=str(path), sha256=sha(path), parameters=field.parameters)
    check_outputs([*([train] if stage != 'evaluate' else []), *([test] if stage != 'train' else [])])
    return dict(configs={'base': base, 'new': new}, data=data, smoke=smoke, note=NOTE,
                cuda=torch.cuda.get_device_name(), baseline=dict(path=str(BASE), sha256=sha(BASE)))


def plot_training(directory, label, output=None):
    legacy_plot_training(directory, f'V3.3.3_m | {label} | monitor: 5 seeds/map', output)
    singular_figures(output or directory)


def cached_baseline(scene, seeds, report):
    meta = json.loads((REFERENCE / 'others/metadata.json').read_text())
    if meta['smoke'] or meta['seeds'] != list(range(1000, 1050)):
        raise ValueError('Reference seed/protocol mismatch')
    if meta['checkpoints']['V3_3_3']['sha256'] != sha(BASE):
        raise ValueError('Reference checkpoint mismatch')
    if meta['fields'][scene]['sha256'] != report['data'][scene]['sha256']:
        raise ValueError('Reference field mismatch')
    saved = load_config(BASE.parent / 'others/config.yaml')
    if saved != report['configs']['base']:
        raise ValueError('Reference config mismatch')
    source = REFERENCE / f'others/{scene}/V3_3_3'
    files = [source / f for f in ('eval_metrics.csv', 'trajectories.csv')]
    rows, paths = [[r for r in typed_rows(p) if int(r['seed']) in seeds] for p in files]
    if len(rows) != len(seeds) * 72 or len(paths) != len(seeds) * 70:
        raise ValueError('Reference record count mismatch')
    audit_scene({METHODS[0]: rows}, {METHODS[0]: paths}, seeds)
    return rows, paths, dict(reused=True, files={str(p): sha(p) for p in files})


def summarize(test, smoke):
    rows = []
    for i in range(1, 6):
        rows.extend(dict(map=f'test{i}', **r) for r in typed_rows(test / f'test{i}/others/summary.csv') if r['step'] == 70)
    for method in METHODS:
        selected = [r for r in rows if r['method'] == method]
        rows.append(dict(map='five_map_mean', method=method, step=70, runs=selected[0]['runs'],
                         **{k: float(np.mean([r[k] for r in selected])) for k in selected[0] if k.endswith(('_mean', '_std'))}))
    write_csv(test / 'others/summary_70.csv', rows)
    lines = ['# V3.3.3 mask 单因素对照' + ('（冒烟，非正式结果）' if smoke else ''), '', NOTE, '',
             '| 地图 | 方法 | RMSE (nT) | Trr (%) | 重复 |', '|---|---|---|---|---|']
    for row in rows:
        cells = [f"{row[k+'_mean']:.4f} ± {row[k+'_std']:.4f}" for k in ('rmse', 'trr', 'repeated_visits')]
        lines.append(f"| {row['map']} | {row['method']} | " + ' | '.join(cells) + ' |')
    lines += ['', '## 相对 V3.3.3 的变化', '']
    for scene in [*(f'test{i}' for i in range(1, 6)), 'five_map_mean']:
        old, new = [next(r for r in rows if r['map'] == scene and r['method'] == m) for m in METHODS]
        lines.append(f"- {scene}：RMSE {new['rmse_mean']-old['rmse_mean']:+.4f} nT；Trr {new['trr_mean']-old['trr_mean']:+.4f} 个百分点；重复 {new['repeated_visits_mean']-old['repeated_visits_mean']:+.4f}。")
    path = test / 'table/report.md'; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')


def report_scene(scene, runs, paths, target, seeds, smoke):
    audit_scene(runs, paths, seeds)
    for method in METHODS:
        if len(runs[method]) != len(seeds)*72 or len(paths[method]) != len(seeds)*70:
            raise ValueError('Unexpected evaluation record counts')
        write_csv(target / f'others/{method}/eval_metrics.csv', runs[method])
        write_csv(target / f'others/{method}/trajectories.csv', paths[method])
    summary = scene_report(scene, runs, paths, target, seeds, smoke, methods=METHODS,
                           note=NOTE, weight_report=True, sample_label=f'{len(seeds)} paired seeds/map')
    for method in METHODS:
        write_csv(target / f'others/{method}/summary.csv', [r for r in summary if r['method'] == method])
    singular_figures(target)
    audit_summary(target, runs)
    write_json(target / 'others/acceptance.json', dict(passed=True, seeds=seeds, steps=70))


def evaluate_pair(train, test, report, smoke):
    audit_training(train, 'V3_3_3_m', smoke, expected_config=report['configs']['new'])
    new, payload = load_policy(train / 'checkpoint.pt')
    if not new.action_mask_enabled or not all(torch.isfinite(p).all() for p in new.network.parameters()):
        raise ValueError('Invalid masked checkpoint')
    base, base_payload = load_policy(BASE)
    seeds = [1000, 1001] if smoke else list(range(1000, 1050))
    empty_output(test)
    reuse = {}
    for i in range(1, 6):
        scene = f'test{i}'; field = FieldModel.load(ROOT / f'data/models/{scene}.pkl')
        runs = {}; paths = {}
        try:
            runs[METHODS[0]], paths[METHODS[0]], reuse[scene] = cached_baseline(scene, seeds, report)
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            c = deepcopy(base_payload['config']); c['field']['test_scenario'] = scene; c['evaluation']['seeds'] = seeds
            runs[METHODS[0]], paths[METHODS[0]], _ = evaluate(field, base.scaler, c, base, episode=10000)
            reuse[scene] = dict(reused=False, reason=str(exc))
        c = deepcopy(payload['config']); c['field']['test_scenario'] = scene; c['evaluation']['seeds'] = seeds
        runs[METHODS[1]], paths[METHODS[1]], _ = evaluate(field, new.scaler, c, new, episode=payload['progress']['episode'])
        report_scene(scene, runs, paths, test / scene, seeds, smoke)
        LOG.info('Paired evaluation completed: %s', scene)
    write_json(test / 'others/metadata.json', dict(**report, seeds=seeds, reference=reuse,
               checkpoints={METHODS[0]: report['baseline'], METHODS[1]: dict(path=str(train / 'checkpoint.pt'), sha256=sha(train / 'checkpoint.pt'))}))
    summarize(test, smoke)
    write_json(test / 'others/acceptance.json', dict(passed=True, smoke=smoke, maps=5, methods=METHODS,
               seeds=seeds, trajectories_per_method=5*len(seeds), steps=70))


def replot(train, test, output, smoke):
    check_outputs([output]); output = Path(output)
    plot_training(train, 'SMOKE' if smoke else 'Formal', output / 'main')
    meta = json.loads((test / 'others/metadata.json').read_text())
    if meta['smoke'] != smoke:
        raise ValueError('Plot mode mismatch')
    target = output / 'comparison'
    for i in range(1, 6):
        scene = f'test{i}'; source = test / scene
        runs = {m: typed_rows(source / f'others/{m}/eval_metrics.csv') for m in METHODS}
        paths = {m: typed_rows(source / f'others/{m}/trajectories.csv') for m in METHODS}
        report_scene(scene, runs, paths, target / scene, meta['seeds'], smoke)
        for relative in ('others/summary.csv', 'table/comparison.md'):
            if (source / relative).read_bytes() != (target / scene / relative).read_bytes():
                raise ValueError('CSV replot differs from original')
    summarize(target, smoke)
    write_json(output / 'acceptance.json', dict(passed=True, csv_only=True, tables_identical=True))


def run(stage, train_root, test_root, output=None, smoke_results=False):
    smoke = stage == 'smoke' or smoke_results
    train, test = directories(train_root, test_root, smoke)
    if stage == 'plot':
        if output is None:
            raise ValueError('plot requires --output')
        return replot(train, test, output, smoke)
    report = preflight(train, test, stage, smoke)
    if stage == 'preflight':
        print(f"Preflight passed: V3.3.3 mask-only; CUDA {report['cuda']}; 10 maps.")
        return report
    protected = protect_history()
    protected.update({v['path']: v['sha256'] for v in report['data'].values()})
    started = time.monotonic()
    try:
        if stage in ('smoke', 'train'):
            train_variant(report['configs']['new'], train, smoke, plotter=plot_training)
            audit_training(train, 'V3_3_3_m', smoke, expected_config=report['configs']['new'])
            metadata = json.loads((train / 'others/metadata.json').read_text())
            write_json(train / 'others/metadata.json', dict(metadata, note=NOTE, preflight=report))
        if stage in ('smoke', 'evaluate'):
            evaluate_pair(train, test, report, smoke)
    except Exception as exc:
        LOG.exception('V3.3.3_m stage failed')
        for path in ([train, test] if stage == 'smoke' else [train if stage == 'train' else test]):
            if path.exists():
                write_json(path / 'others/pipeline_failure.json', dict(stage=stage, error=repr(exc)))
        raise
    finally:
        verify_hashes(protected)
    for path in ([train, test] if stage == 'smoke' else [train if stage == 'train' else test]):
        write_json(path / 'others/execution.json', dict(passed=True, stage=stage, history_verified=True,
                   seconds=time.monotonic()-started, protected_sha256=protected))
