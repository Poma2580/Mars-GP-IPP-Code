"""Validate current-map version evaluations and write traceable numeric summaries."""
import _bootstrap
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from mars_ipp.envs.actions import ACTIONS
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.training.checkpoint import read_checkpoint
from mars_ipp.utils.config import load_config, merge
from mars_ipp.utils.logger import write_csv, write_json
from mars_ipp.visualization.learning_curves import read_csv

ROOT = _bootstrap.ROOT
METHODS = ('V0', 'V0.1', 'V1', 'V1.1', 'V2', 'V2-GlobalOnly')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_directories(root, scene):
    comparison = ROOT / 'results/comparison'
    return {
        'V0': [root / scene / 'v0'],
        'V0.1': [root / scene / 'v0_1'],
        'V1': [comparison / 'bnorm_no_mh_uncertain_rbf' / scene / 'ours'],
        'V1.1': [root / scene / 'v1_1'],
        **{label: [comparison / 'bnorm_no_mh_local_global_3x3_rbf' / scene / method / f'seed_{seed}'
                   for seed in (0, 1, 2)]
           for label, method in (('V2', 'v2'), ('V2-GlobalOnly', 'v2_global_only'))},
    }


def validate_evaluation(directory, cfg, fields, scene, method, training_seed):
    config = yaml.safe_load((directory / 'config.yaml').read_text())
    meta = json.loads((directory / 'metadata.json').read_text())
    assert config['env'] == cfg['env'], f'Environment mismatch: {directory}'
    assert config['evaluation']['seeds'] == list(range(1000, 1050))
    assert config['seed'] == training_seed and config['field']['test_scenario'] == scene
    assert meta['fields'] == fields, f'Field mismatch: {directory}'
    checkpoint = Path(meta['checkpoint'])
    assert digest(checkpoint) == meta['checkpoint_sha256']
    payload = read_checkpoint(checkpoint)
    assert payload['progress'] == {'episode': 10000, 'steps': 700000, 'completed': True}
    assert meta['scaler'] == payload['scaler']
    assert meta['checkpoint_training_config'] == payload['config']
    if method in ('V0', 'V0.1'):
        assert meta['preserve_checkpoint_scaler']
        assert payload['config']['env']['grid_size'] == 15
        assert payload['scaler']['coordinate_scale'] == 14
        assert config == merge(payload['config'], {'env': cfg['env'], 'evaluation': {'seeds': list(range(1000, 1050))},
                                                   'field': {'test_scenario': scene}})
        assert payload['config']['ppo'].get('observation_mode', 'full') == ('full' if method == 'V0' else 'position_only')
    rows = read_csv(directory / 'eval_metrics.csv')
    paths = read_csv(directory / 'trajectories.csv')
    assert len(rows) == 3600 and len(paths) == 3500
    assert {int(r['seed']) for r in rows} == set(range(1000, 1050))
    assert {int(r['seed']) for r in paths} == set(range(1000, 1050))
    expected_stages = [('prior', 0), ('reset', 0)] + [('step', s) for s in range(1, 71)]
    for seed in range(1000, 1050):
        selected = [r for r in rows if int(r['seed']) == seed]
        assert [(r['stage'], int(r['step'])) for r in selected] == expected_stages
        start = np.random.default_rng(seed).integers(0, 16, 2)
        assert all((int(r['start_i']), int(r['start_j'])) == tuple(start) for r in selected)
        assert all(int(r['episode']) == 10000 for r in selected)
        for row in selected:
            values = np.array([float(row[k]) for k in ('rmse', 'trr', 'trace_current', 'trace_prior')])
            assert np.isfinite(values).all() and (values >= 0).all()
            assert np.isclose(float(row['trr']), float(row['trace_current']) / float(row['trace_prior']) * 100)
            assert float(row['online_noise_variance']) == 1e-6
            assert int(row['samples']) == int(row['unique_positions'])
        trajectory = [r for r in paths if int(r['seed']) == seed]
        assert [int(r['step']) for r in trajectory] == list(range(1, 71))
        position = start.copy()
        visited = {tuple(position)}
        for row, metric in zip(trajectory, selected[2:]):
            action = int(row['action'])
            assert 0 <= action < 8
            candidate = position + ACTIONS[action]
            if ((candidate >= 0) & (candidate < 16)).all():
                position = candidate
            assert tuple(position) == (int(row['i']), int(row['j']))
            visited.add(tuple(position))
            assert int(metric['unique_positions']) == len(visited)
    return [{**r, 'method': method, 'training_seed': training_seed, 'scenario': scene} for r in rows]


def audit(root):
    root = Path(root).resolve()
    cfg = load_config(ROOT / 'configs/main/bnorm_no_mh.yaml')
    fields = {name: {'sha256': field.sha256, 'parameters': field.parameters}
              for name, field in prepare_fields(cfg).items()}
    training = ROOT / 'results/main/bnorm_no_mh_rbf/seed_0'
    payload = read_checkpoint(training / 'checkpoint.pt')
    assert payload['config'] == cfg
    assert payload['input_dim'] == 5 and payload['scaler']['mode'] == 'signed_extrema_bnorm'
    status = json.loads((training / 'run_status.json').read_text())
    assert status['completed'] and status['episode'] == 10000 and status['steps'] == 700000
    assert status['updates'] == 350 and status['pending_rollout'] == 0
    train_rows = read_csv(training / 'train_metrics.csv')
    assert [int(r['episode']) for r in train_rows] == list(range(1, 10001))
    assert [int(r['steps']) for r in train_rows] == list(range(70, 700001, 70))
    updates = read_csv(training / 'updates.csv')
    assert len(updates) == 350 and all(int(r['samples']) == 2000 for r in updates)
    monitor = read_csv(training / 'eval_endpoints.csv')
    assert [(int(r['episode']), int(r['seed'])) for r in monitor] == [
        (episode, seed) for episode in range(200, 10001, 200) for seed in range(1000, 1050)]
    assert all(r['stage'] == 'step' and int(r['step']) == 70 for r in monitor)
    manifest = json.loads((root / 'original_artifact_hashes.json').read_text())
    assert all(digest(ROOT / path) == value for path, value in manifest.items()), 'Original artifacts changed'
    summaries, curves, sources, endpoints = [], [], [], []
    for scene in ('zhurong', 'perseverance'):
        scene_rows = []
        for method, directories in source_directories(root, scene).items():
            rows = []
            for index, directory in enumerate(directories):
                seed = index if len(directories) == 3 else 0
                rows.extend(validate_evaluation(directory, cfg, fields, scene, method, seed))
                sources.append({'scenario': scene, 'method': method, 'training_seed': seed,
                                'directory': str(directory), 'metrics_sha256': digest(directory / 'eval_metrics.csv')})
            final = [r for r in rows if r['stage'] == 'step' and int(r['step']) == 70]
            summary = {'scenario': scene, 'method': method, 'runs': len(final)}
            for metric in ('rmse', 'trr'):
                values = [float(r[metric]) for r in final]
                summary.update({f'{metric}_mean': float(np.mean(values)), f'{metric}_std': float(np.std(values, ddof=0))})
            summaries.append(summary)
            endpoints.extend({key: r[key] for key in ('scenario', 'method', 'training_seed', 'seed', 'start_i', 'start_j', 'rmse', 'trr')}
                             for r in final)
            for step in range(71):
                selected = [r for r in rows if int(r['step']) == step and r['stage'] == ('reset' if step == 0 else 'step')]
                assert len(selected) == len(final)
                entry = {'scenario': scene, 'method': method, 'step': step, 'runs': len(selected)}
                for metric in ('rmse', 'trr'):
                    values = [float(r[metric]) for r in selected]
                    entry.update({f'{metric}_mean': float(np.mean(values)), f'{metric}_std': float(np.std(values, ddof=0))})
                curves.append(entry)
            for metric in ('rmse', 'trr'):
                assert summary[f'{metric}_mean'] == curves[-1][f'{metric}_mean']
                assert summary[f'{metric}_std'] == curves[-1][f'{metric}_std']
            scene_rows.extend(rows)
        # Reward components differ by version; keep their union in the combined CSV.
        columns = list(dict.fromkeys(key for row in scene_rows for key in row))
        write_csv(root / scene / 'combined_eval_metrics.csv',
                  [{key: row.get(key, '') for key in columns} for row in scene_rows])
        v1_1_final = [r for r in scene_rows if r['method'] == 'V1.1' and r['stage'] == 'step' and int(r['step']) == 70]
        if scene == 'perseverance':
            last_monitor = [r for r in monitor if int(r['episode']) == 10000]
            for a, b in zip(v1_1_final, last_monitor):
                assert a['seed'] == b['seed']
                for metric in ('rmse', 'trr'):
                    assert np.isclose(float(a[metric]), float(b[metric]), rtol=1e-10, atol=1e-10)
    write_csv(root / 'summary.csv', summaries)
    write_csv(root / 'per_episode_endpoints.csv', endpoints)
    write_csv(root / 'step_statistics.csv', curves)
    write_json(root / 'sources.json', {'sources': sources, 'std': 'population, ddof=0',
               'trr': 'trace_current / trace_prior * 100; lower is better',
               'v0_protocol': '15x15-trained policies evaluated on 16x16; exact training scalers retained',
               'v2_runs': '3 training seeds x 50 paired test seeds; 150 episodes, 50 distinct starts'})
    write_json(root / 'validation.json', {'passed': True, 'original_artifacts_unchanged': len(manifest),
               'v1_1_training': status, 'evaluation_groups': len(sources), 'new_evaluation_episodes': 300,
               'new_metric_rows': 21600, 'new_action_rows': 21000, 'paired_starts': True,
               'stages_and_trajectories_complete': True, 'curve_endpoints_match_tables': True,
               'final_training_monitor_matches_formal_evaluation': True})
    for row in summaries:
        print(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT / 'results/main/version_comparison_current'))
    audit(parser.parse_args().root)
