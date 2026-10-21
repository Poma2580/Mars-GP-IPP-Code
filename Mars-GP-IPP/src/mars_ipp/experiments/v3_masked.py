"""Independent masked variants with paired evaluation and artifact audits."""
from collections import Counter
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
from mars_ipp.utils.config import ROOT, load_config
from mars_ipp.utils.seed import seed_all
from mars_ipp.utils.logger import write_csv, write_json
from mars_ipp.visualization.learning_curves import read_csv
from .v3_pipeline import (METRICS, empty_output, load_policy, protect_history,
                          scene_report, sha, train_variant, validate_scene, verify_hashes)
from .v3_variants import NEW_VERSIONS, TRAIN_MAPS, VariantEnv
from .v3_weights import plot_training as legacy_plot_training

LOG = logging.getLogger(__name__)
LABELS = {'v3_3_4': 'V3.3.4', 'v3_5': 'V3.5'}
SLUGS = {'v3_3_4': 'v_3.3.4', 'v3_5': 'v_3.5'}
HISTORICAL = {
    'V3.3.3': ROOT / 'results/main/v3_3_weight_train/V3_3_3/checkpoint.pt',
    'V3.4.1': ROOT / 'results/main/v3_4_weight_train/V3_4_1/checkpoint.pt',
}
METHODS = {'v3_3_4': ('GP-Trace', 'V3.3.3', 'V3.3.4'),
           'v3_5': ('GP-Trace', 'V3.3.4', 'V3.4.1', 'V3.5')}
NOTE = ('Bmax comes from the known per-map reference-field prior, not unknown online truth. '
        'Historical PPO policies retain original unmasked behavior. '
        'V3.3.4 changes both masking and repeat weight; V3.5 adds both B state and MH reward. '
        'Results cannot be attributed to a single factor. Population SD ddof=0; lower RMSE/Trr is better.')


def directories(train_root, test_root, smoke):
    return ({v: Path(train_root) / f'{SLUGS[v]}_{"smoke" if smoke else "train"}' for v in NEW_VERSIONS},
            {v: Path(test_root) / f'{SLUGS[v]}_{"smoke_test" if smoke else "test"}' for v in NEW_VERSIONS})


def check_outputs(paths):
    paths = [Path(p).resolve() for p in paths]
    if len(set(paths)) != len(paths) or any(a in b.parents for a in paths for b in paths if a != b):
        raise ValueError('Output paths must be distinct and non-nested')
    for path in paths:
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f'Refuse to overwrite nonempty output: {path}')


def preflight(train_root, test_root, smoke=True, check_train=True, check_test=True):
    configs = {v: load_config(ROOT / f'configs/main/{v}.yaml') for v in NEW_VERSIONS}
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    data = {}
    for kind in ('train', 'test'):
        for i in range(1, 6):
            name = f'{kind}{i}'; path = ROOT / f'data/models/{name}.pkl'
            field = FieldModel.load(path)
            if not np.isfinite(field.truth(np.array([[0., 0.], [30., 30.]]))).all():
                raise ValueError(f'Nonfinite field: {path}')
            data[name] = {'path': str(path), 'sha256': sha(path), 'parameters': field.parameters}
    history = {}
    for label, path in HISTORICAL.items():
        agent, payload = load_policy(path)
        expected = load_config(ROOT / f'configs/main/{"V3_3_3" if label == "V3.3.3" else "V3_4_1"}.yaml')
        if (payload['config'] != expected or agent.action_mask_enabled
                or payload['progress'] != {'episode': 10000, 'steps': 700000, 'completed': True}):
            raise ValueError(f'Invalid historical checkpoint: {path}')
        field = FieldModel.load(ROOT / 'data/models/train1.pkl')
        obs, _ = VariantEnv(field, agent.scaler, expected, 'terminal').reset(seed=1000)
        agent.act(obs)
        history[label] = {'path': str(path), 'sha256': sha(path), 'progress': payload['progress']}
    train_dirs, test_dirs = directories(train_root, test_root, smoke)
    check_outputs([*(train_dirs.values() if check_train else []), *(test_dirs.values() if check_test else [])])
    return {'configs': configs, 'data': data, 'historical': history,
            'cuda': torch.cuda.get_device_name(), 'smoke': smoke, 'note': NOTE}


def singular_figures(directory):
    old = Path(directory) / 'figures'
    if old.exists():
        old.rename(Path(directory) / 'figure')


def plot_training(directory, label, output=None):
    target = Path(output or directory)
    version = json.loads((Path(directory) / 'others/metadata.json').read_text())['scaler']['mode']
    label += ' | ' + ('V3.5' if version == 'gp_trace_b_14d' else 'V3.3.4')
    legacy_plot_training(directory, label + ' | monitor: 5 seeds/map; raw Trace scales differ', output=target)
    singular_figures(target)


def audit_training(directory, version, smoke, expected_config=None):
    directory = Path(directory); aux = directory / 'others'
    _, payload = load_policy(directory / 'checkpoint.pt')
    expected = deepcopy(expected_config) if expected_config is not None else load_config(ROOT / f'configs/main/{version}.yaml')
    if smoke:
        expected['training'].update(episodes=30, checkpoint_every=10, eval_every=10)
    if payload['config'] != expected:
        raise ValueError('Unexpected training configuration')
    episodes = 30 if smoke else 10000
    if payload['progress'] != {'episode': episodes, 'steps': episodes * 70, 'completed': True}:
        raise ValueError('Incomplete training checkpoint')
    rows = read_csv(aux / 'train_metrics.csv'); steps = read_csv(aux / 'train_steps.csv')
    updates = read_csv(aux / 'updates.csv'); monitor = read_csv(aux / 'eval_endpoints.csv')
    components = ('gp', 'rep', 'mh') if version == 'v3_5' else ('gp', 'rep')
    if [int(r['episode']) for r in rows] != list(range(1, episodes + 1)):
        raise ValueError('Missing training episodes')
    if Counter(r['map'] for r in rows) != Counter(dict.fromkeys(TRAIN_MAPS, episodes // 5)):
        raise ValueError('Unbalanced maps')
    if len(steps) != episodes * 70 or sum(int(r['samples']) for r in updates) != len(steps):
        raise ValueError('Incomplete steps or rollout updates')
    for r in [*rows, *steps, *monitor, *updates]:
        for value in r.values():
            try:
                number = float(value)
            except (ValueError, TypeError):
                continue
            if not np.isfinite(number):
                raise ValueError('Nonfinite training log')
    # Validate whole columns once instead of making millions of scalar assertions.
    for records in (rows, steps):
        reward = np.asarray([float(r['reward']) for r in records])
        weighted = np.asarray([[float(r[f'weighted_r_{k}']) for k in components] for r in records])
        raw = np.asarray([[float(r[f'r_{k}']) for k in components] for r in records])
        np.testing.assert_allclose(reward, weighted.sum(axis=1), atol=1e-10)
        np.testing.assert_allclose(weighted, raw * expected['reward']['weights'], atol=1e-12)
        if expected_config is not None and records is steps:
            if not np.isin(raw[:, 1], [-1., 0.]).all():
                raise ValueError('Invalid repeat indicator')
            repeated = raw[:, 1] == -1.
            if np.any(raw[repeated][:, [i for i, name in enumerate(components) if name != 'rep']] != 0):
                raise ValueError('Repeated visit has GP/MH reward')
        if any(int(r['outside_actions']) != 0 for r in records):
            raise ValueError('Outside action in masked training')
    steps_by_episode = {}
    for step in steps:
        steps_by_episode.setdefault(step['episode'], []).append(step)
    for r in rows:
        ep_steps = steps_by_episode.get(r['episode'], [])
        if [int(s['step']) for s in ep_steps] != list(range(1, 71)) or any(s['map'] != r['map'] for s in ep_steps):
            raise ValueError('Incomplete episode steps')
        for k in components:
            np.testing.assert_allclose(float(r[f'r_{k}']), sum(float(s[f'r_{k}']) for s in ep_steps), atol=1e-10)
        np.testing.assert_allclose(float(r['r_gp']), float(r['raw_trace_sum']) / float(r['trace_prior']), rtol=2e-7, atol=1e-7)
        if int(r['unique_positions']) + int(r['repeated_visits']) != 71:
            raise ValueError('Training sample accounting failed')
    interval = 10 if smoke else 200
    expected_monitor = {(ep, scene, seed) for ep in range(interval, episodes + 1, interval)
                        for scene in TRAIN_MAPS for seed in expected['training']['monitor_seeds']}
    actual = [(int(r['episode']), r['map'], int(r['seed'])) for r in monitor]
    if len(actual) != len(expected_monitor) or set(actual) != expected_monitor or any(int(r['outside_actions']) for r in monitor):
        raise ValueError('Missing or invalid monitor records')
    status = json.loads((aux / 'run_status.json').read_text())
    if not status['completed'] or status['pending_rollout']:
        raise ValueError('Unfinished training')
    for name in ('training', 'training_evaluation'):
        if not (directory / f'figure/{name}.png').is_file():
            raise ValueError('Missing training figure')
    result = {'passed': True, 'smoke': smoke, 'episodes': episodes, 'steps': len(steps),
              'monitor_records': len(monitor), 'checkpoint_sha256': sha(directory / 'checkpoint.pt')}
    write_json(aux / 'summary.json', result); write_json(aux / 'acceptance.json', result)
    return result


def audit_scene(runs, paths, seeds):
    validate_scene(runs, paths, seeds, tuple(runs))
    for method, rows in runs.items():
        for seed in seeds:
            selected = [r for r in rows if int(r['seed']) == seed and r['stage'] != 'prior']
            reset = selected[0]; position = np.array([reset['start_i'], reset['start_j']], dtype=int)
            visited = {tuple(position)}; repeats = outside = 0
            trajectory = [r for r in paths[method] if int(r['seed']) == seed]
            for row, path in zip(selected[1:], trajectory):
                next_position = position + ACTIONS[int(path['action'])]
                if np.any(next_position < 0) or np.any(next_position >= 16):
                    outside += 1
                else:
                    repeats += int(tuple(next_position) in visited)
                    position = next_position; visited.add(tuple(position))
                if list(position) != [int(path['i']), int(path['j'])]:
                    raise ValueError('Trajectory action/position mismatch')
                if [len(visited), repeats, outside] != [int(row[k]) for k in ('unique_positions', 'repeated_visits', 'outside_actions')]:
                    raise ValueError('Sample accounting mismatch')
                if method in ('V3.3.4', 'V3.5', 'V3.3.3_m'):
                    if outside:
                        raise ValueError('Masked policy outside action')
                    components = ('gp', 'rep', 'mh') if method == 'V3.5' else ('gp', 'rep')
                    np.testing.assert_allclose(path['reward'], sum(row[f'weighted_r_{k}'] for k in components), atol=1e-10)
                    if not row['sampled'] and any(row[f'r_{k}'] != 0 for k in ('gp', 'mh') if f'r_{k}' in row):
                        raise ValueError('Unsampled GP/MH reward')
            if method in ('V3.3.4', 'V3.5', 'V3.3.3_m'):
                np.testing.assert_allclose(sum(r['r_gp'] for r in selected[1:]),
                    (reset['trace_current'] - selected[-1]['trace_current']) / reset['trace_prior'], rtol=2e-7, atol=1e-7)


def audit_summary(directory, runs):
    rows = read_csv(Path(directory) / 'others/summary.csv')
    if len(rows) != 3 * len(runs):
        raise ValueError('Incomplete summary')
    for item in rows:
        source = [r for r in runs[item['method']] if r['stage'] == 'step' and int(r['step']) == int(item['step'])]
        if int(item['runs']) != len(source):
            raise ValueError('Summary run count mismatch')
        for metric in METRICS:
            values = [float(r[metric]) for r in source]
            np.testing.assert_allclose([float(item[metric + '_mean']), float(item[metric + '_std'])],
                                       [np.mean(values), np.std(values, ddof=0)], atol=1e-12)
    table = (Path(directory) / 'table/comparison.md').read_text()
    for item in rows:
        cells = [f"{float(item[k + '_mean']):.4f} ± {float(item[k + '_std']):.4f}" for k in METRICS]
        if '| ' + item['method'] + ' | ' + ' | '.join(cells) + ' |' not in table:
            raise ValueError('Comparison table differs from summary')
    for budget in (30, 50, 70):
        if f'预算{budget}步表现' not in table:
            raise ValueError('Missing budget section')
    for name in ('rmse', 'trr', 'trajectories_seed_1000'):
        if not (Path(directory) / f'figure/{name}.png').is_file():
            raise ValueError('Missing comparison figure')


def evaluate_new(train_dirs, test_dirs, report, smoke):
    sources = {**HISTORICAL, **{LABELS[v]: p / 'checkpoint.pt' for v, p in train_dirs.items()}}
    for version, path in train_dirs.items():
        audit_training(path, version, smoke)
    agents = {label: load_policy(path) for label, path in sources.items()}
    seeds = [1000, 1001] if smoke else list(range(1000, 1050))
    for path in test_dirs.values():
        empty_output(path)
    provenance = {label: {'path': str(path), 'sha256': sha(path)} for label, path in sources.items()}
    for i in range(1, 6):
        scene = f'test{i}'; field = FieldModel.load(ROOT / f'data/models/{scene}.pkl')
        baseline = deepcopy(report['configs']['v3_3_4'])
        baseline['field']['test_scenario'] = scene; baseline['evaluation']['seeds'] = seeds
        runs = {}; paths = {}
        runs['GP-Trace'], paths['GP-Trace'], _ = evaluate(field, GPTraceScaler(), baseline, algorithm='gp_trace')
        for label, (agent, payload) in agents.items():
            cfg = deepcopy(payload['config']); cfg['field']['test_scenario'] = scene; cfg['evaluation']['seeds'] = seeds
            runs[label], paths[label], _ = evaluate(field, agent.scaler, cfg, agent, episode=payload['progress']['episode'])
        for version, methods in METHODS.items():
            output = test_dirs[version] / scene
            selected = {m: runs[m] for m in methods}; trajectories = {m: paths[m] for m in methods}
            audit_scene(selected, trajectories, seeds)
            for method in methods:
                write_csv(output / f'others/{method}/eval_metrics.csv', runs[method])
                write_csv(output / f'others/{method}/trajectories.csv', paths[method])
            summary = scene_report(scene, selected, trajectories, output, seeds, smoke,
                                   methods=methods, note=NOTE + f' Seeds={seeds}.', weight_report=True, sample_label=f'{len(seeds)} seeds')
            for method in methods:
                write_csv(output / f'others/{method}/summary.csv', [r for r in summary if r['method'] == method])
            singular_figures(output)
            saved = {m: read_csv(output / f'others/{m}/eval_metrics.csv') for m in methods}
            audit_summary(output, saved)
            write_json(output / 'others/acceptance.json', {'passed': True, 'seeds': seeds, 'steps': 70})
        LOG.info('Completed paired evaluation %s', scene)
    for output in test_dirs.values():
        write_json(output / 'others/metadata.json', {**report, 'checkpoints': provenance, 'seeds': seeds})
        write_json(output / 'others/acceptance.json', {'passed': True, 'maps': 5, 'seeds': seeds, 'smoke': smoke})


def replot(train_dirs, test_dirs, output, smoke):
    check_outputs([output]); output = Path(output)
    for version, source in train_dirs.items():
        plot_training(source, ('SMOKE - non-formal' if smoke else 'Formal') + f' {LABELS[version]}', output / 'main' / source.name)
    seeds = [1000, 1001] if smoke else list(range(1000, 1050))
    for version, source in test_dirs.items():
        for i in range(1, 6):
            scene = f'test{i}'; methods = METHODS[version]
            runs = {m: read_csv(source / scene / f'others/{m}/eval_metrics.csv') for m in methods}
            paths = {m: read_csv(source / scene / f'others/{m}/trajectories.csv') for m in methods}
            target = output / 'comparison' / source.name / scene
            scene_report(scene, runs, paths, target, seeds, smoke, methods=methods, note=NOTE, weight_report=True, sample_label=f'{len(seeds)} seeds')
            singular_figures(target); audit_summary(target, runs)


def run(stage, train_root, test_root, output=None, smoke_results=False):
    smoke = stage in ('smoke', 'preflight') or smoke_results
    train_dirs, test_dirs = directories(train_root, test_root, smoke)
    if stage == 'plot':
        return replot(train_dirs, test_dirs, output, smoke)
    seed_all(0)
    report = preflight(train_root, test_root, smoke, check_train=stage != 'evaluate', check_test=stage != 'train')
    if stage == 'preflight':
        print(json.dumps(report, indent=2, default=str)); return report
    protected = protect_history(); started = time.monotonic()
    try:
        if stage in ('smoke', 'train'):
            for version, directory in train_dirs.items():
                LOG.info('Training %s smoke=%s', version, smoke)
                train_variant(report['configs'][version], directory, smoke, plotter=plot_training)
                audit_training(directory, version, smoke)
                metadata = directory / 'others/metadata.json'
                payload = json.loads(metadata.read_text()); payload.update(note=NOTE, preflight=report)
                write_json(metadata, payload)
        if stage in ('smoke', 'evaluate'):
            evaluate_new(train_dirs, test_dirs, report, smoke)
    except Exception as exc:
        for directory in (*train_dirs.values(), *test_dirs.values()):
            if directory.exists():
                write_json(directory / 'others/pipeline_failure.json', {'error': repr(exc), 'stage': stage,
                           'seconds': time.monotonic() - started})
        raise
    finally:
        verify_hashes(protected)
    for directory in (train_dirs.values() if stage == 'train' else test_dirs.values()):
        write_json(directory / 'others/execution.json', {'passed': True, 'stage': stage,
                   'seconds': time.monotonic() - started, 'history_verified': True,
                   'protected_sha256': protected})
