"""Evaluate frozen seed-0 policies and four baselines on test2 through test5."""
import _bootstrap
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import yaml
from mars_ipp.agents.baselines.msr import Msr
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.envs.actions import ACTIONS
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.envs.observations import SceneObservationScaler, SCALER_V2, scaler_from_dict, make_scaler
from mars_ipp.evaluation.evaluator import evaluate, env_config
from mars_ipp.field.field_model import FieldModel
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.training.checkpoint import read_checkpoint, restore_checkpoint, scalers_equivalent
from mars_ipp.utils.config import ROOT, load_config, merge, validate_output_path
from mars_ipp.utils.logger import write_csv, write_json
from mars_ipp.utils.seed import seed_all
from mars_ipp.visualization.learning_curves import read_csv

MAPS = ('test2', 'test3', 'test4', 'test5')
METHODS = ('gp_greedy', 'gp_trace', 'msr', 'cover', 'v2', 'v2_global_only')
LABELS = dict(zip(METHODS, ('GP-Entropy Active Learning', 'GP-Trace Active Learning',
                          'Msr', 'Cover', 'V2', 'V2-GlobalOnly')))
CHECKPOINTS = {method: ROOT / 'results/main' / experiment / 'seed_0/checkpoint.pt'
               for method, experiment in (('msr', 'msr'), ('v2', 'bnorm_no_mh_local_global_3x3_rbf'),
                                         ('v2_global_only', 'bnorm_no_mh_local_global_3x3_rbf_no_local'))}
DEFAULT_DATA = ROOT.parent / 'Mars_Env_GP/output/10datasets'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_agent(method):
    checkpoint = CHECKPOINTS[method]
    payload = read_checkpoint(checkpoint)
    cfg = payload['config']
    if cfg['seed'] != 0 or payload['progress'] != {'episode': 10000, 'steps': 700000, 'completed': True}:
        raise ValueError('Require completed seed-0, 10000-episode checkpoint')
    expected_mode = 'coordinates_only' if method == 'msr' else 'full_uncertainty_local_global_3x3'
    if cfg['ppo']['observation_mode'] != expected_mode:
        raise ValueError('Checkpoint observation mode differs from requested method')
    if method != 'msr' and cfg['reward']['weights'] != ([.25]*4 if method == 'v2' else [0., .25, .25, .25]):
        raise ValueError('Checkpoint reward variant differs from requested method')
    original = scaler_from_dict(payload['scaler'])
    if not scalers_equivalent(payload['scaler'], make_scaler(prepare_fields(cfg), cfg).to_dict()):
        raise ValueError('Original training fields no longer match checkpoint normalization')
    agent = (Msr if method == 'msr' else PPO)(cfg['ppo'], original)
    restore_checkpoint(payload, agent)
    return agent, payload


def evaluation_setup(method, scene, field, seeds):
    base = load_config(ROOT / 'configs/comparison/bnorm_baselines.yaml')
    agent, payload = frozen_agent(method) if method in CHECKPOINTS else (None, None)
    config = merge(payload['config'] if payload else base,
                   {'env': base['env'], 'evaluation': {'seeds': list(seeds), 'cover_random_start': True},
                    'field': {'test_scenario': scene}})
    config['field']['models'] = {scene: field.source}
    if method == 'msr':
        scaler = agent.scaler
    else:
        scaler = SceneObservationScaler.from_fields({scene: field}, 16, 2., scene)
        if payload:
            scaler.mode = SCALER_V2
            scaler.coordinate_scale = agent.scaler.coordinate_scale
    return config, agent, scaler, payload


def validate_rows(rows, paths, seeds, method, episode, field):
    if len(rows) != 72 * len(seeds) or len(paths) != 70 * len(seeds):
        raise ValueError('Incomplete metric or action row counts')
    if {int(r['seed']) for r in rows} != set(seeds) or {int(r['seed']) for r in paths} != set(seeds):
        raise ValueError('Unexpected evaluation seeds')
    stages = [('prior', 0), ('reset', 0)] + [('step', step) for step in range(1, 71)]
    grid = np.array([(i, j) for i in range(16) for j in range(16)])
    truth = field.truth(grid * 2).reshape(16, 16, 3)
    for seed in seeds:
        selected = [r for r in rows if int(r['seed']) == seed]
        if [(r['stage'], int(r['step'])) for r in selected] != stages:
            raise ValueError('Incomplete or duplicate metric stages')
        start = np.random.default_rng(seed).integers(0, 16, 2)
        for row in selected:
            if (int(row['start_i']), int(row['start_j'])) != tuple(start):
                raise ValueError('Unpaired initial position')
            if row['algorithm'] != method or int(row['episode']) != episode:
                raise ValueError('Incorrect policy identity')
            numbers = np.array([float(row[k]) for k in ('rmse', 'trr', 'trace_prior', 'trace_current')])
            if not np.isfinite(numbers).all() or (numbers < 0).any() or numbers[2] <= 0:
                raise ValueError('Invalid metrics')
            if not np.isclose(numbers[1], numbers[3] / numbers[2] * 100):
                raise ValueError('Inconsistent Trr')
            if float(row['online_noise_variance']) != 1e-6 or int(row['samples']) != int(row['unique_positions']):
                raise ValueError('Online GP protocol mismatch')
        trajectory = [r for r in paths if int(r['seed']) == seed]
        if [int(r['step']) for r in trajectory] != list(range(1, 71)):
            raise ValueError('Incomplete action trajectory')
        position = start.copy()
        visited = {tuple(position)}
        for row, metric in zip(trajectory, selected[2:]):
            action = int(row['action'])
            if not 0 <= action < 8:
                raise ValueError('Invalid action')
            candidate = position + ACTIONS[action]
            if ((candidate >= 0) & (candidate < 16)).all():
                position = candidate
            if tuple(position) != (int(row['i']), int(row['j'])):
                raise ValueError('Trajectory disagrees with action')
            if not np.allclose([float(row[k]) for k in ('Br', 'B2', 'B3')], truth[tuple(position)]):
                raise ValueError('Trajectory measurement differs from map')
            visited.add(tuple(position))
            if int(metric['unique_positions']) != len(visited):
                raise ValueError('Incorrect coverage')


def evaluate_method(method, scene, field, origin, output, seeds):
    seed_all(0)
    config, agent, scaler, payload = evaluation_setup(method, scene, field, seeds)
    env = GridEnv(field, scaler, env_config(config), metrics_mode='terminal')
    checkpoint = CHECKPOINTS.get(method)
    metadata = {'method': LABELS[method], 'algorithm': method, 'scenario': scene,
                'checkpoint': str(checkpoint) if checkpoint else None,
                'checkpoint_sha256': sha256(checkpoint) if checkpoint else None,
                'checkpoint_training_config': payload['config'] if payload else None,
                'training_scaler': payload['scaler'] if payload else None,
                'evaluation_scaler': scaler.to_dict(), 'training_seed': 0 if payload else None,
                'field': {'source': field.source, 'sha256': field.sha256, 'parameters': field.parameters,
                          'origin_degrees': origin, 'coordinate_protocol': 'local 0..30 degrees; pkl already uses local inputs'},
                'm_max': env.m_max, 'c_global': env.c_global,
                'gp_protocol': {'noise_variance': 1e-6, 'unique_samples': True, 'known_map_kernel': True},
                'normalization_protocol': 'known per-map signed extrema for V2; original coordinate scaling for learned policies',
                'versions': {name: importlib.metadata.version(name) for name in ('GPy', 'numpy', 'scipy', 'torch')},
                'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'src/mars_ipp').rglob('*.py'))}}
    output = Path(output)
    episode = 10000 if payload else 0
    if output.exists():
        try:
            if yaml.safe_load((output / 'config.yaml').read_text()) != config:
                raise ValueError('Existing configuration differs')
            if json.loads((output / 'metadata.json').read_text()) != metadata:
                raise ValueError('Existing provenance differs')
            rows, paths = read_csv(output / 'eval_metrics.csv'), read_csv(output / 'trajectories.csv')
            validate_rows(rows, paths, seeds, method, episode, field)
            with np.load(output / 'field_snapshot.npz') as snapshot:
                if not np.isfinite(snapshot['mean']).all() or snapshot['mean'].shape != (961, 3):
                    raise ValueError('Invalid field snapshot')
            print(f'Reuse verified {scene}/{method}', flush=True)
            return rows, paths, output
        except (OSError, ValueError, KeyError, TypeError):
            stem = output.name
            index = 1
            while output.exists():
                output = output.with_name(f'{stem}_rerun_{index}')
                index += 1
    output.mkdir(parents=True, exist_ok=False)
    print(f'Evaluate {scene}/{method}: {len(seeds)} episodes -> {output}', flush=True)
    original_scaler = agent.scaler if agent else None
    weights = {key: value.detach().clone() for key, value in agent.network.state_dict().items()} if agent else None
    algorithm = 'ours' if method.startswith('v2') else method
    rows, paths, snapshot = evaluate(field, scaler, config, agent, algorithm, episode)
    for row in (*rows, *paths):
        row['algorithm'] = method
    if agent:
        if agent.scaler is not original_scaler or any(not value.equal(agent.network.state_dict()[key]) for key, value in weights.items()):
            raise ValueError('Evaluation changed original scaler or weights')
    validate_rows(rows, paths, seeds, method, episode, field)
    write_csv(output / 'eval_metrics.csv', rows)
    write_csv(output / 'trajectories.csv', paths)
    np.savez_compressed(output / 'field_snapshot.npz', **snapshot)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(output / 'metadata.json', metadata)
    return rows, paths, output


def summarize_scene(scene, runs, output):
    summary, curves, endpoints, sources = [], [], [], []
    for method, (rows, paths, directory) in runs.items():
        final = [r for r in rows if r['stage'] == 'step' and int(r['step']) == 70]
        entry = {'scenario': scene, 'algorithm': method, 'method': LABELS[method], 'runs': len(final)}
        for metric in ('rmse', 'trr'):
            values = [float(r[metric]) for r in final]
            entry.update({f'{metric}_mean': float(np.mean(values)), f'{metric}_std': float(np.std(values, ddof=0))})
        summary.append(entry)
        endpoints.extend({'scenario': scene, **{key: r[key] for key in ('algorithm', 'seed', 'episode', 'rmse', 'trr')}} for r in final)
        for step in range(71):
            selected = [r for r in rows if int(r['step']) == step and r['stage'] == ('reset' if step == 0 else 'step')]
            point = {'scenario': scene, 'algorithm': method, 'method': LABELS[method], 'step': step, 'runs': len(selected)}
            for metric in ('rmse', 'trr'):
                values = [float(r[metric]) for r in selected]
                point.update({f'{metric}_mean': float(np.mean(values)), f'{metric}_std': float(np.std(values, ddof=0))})
            curves.append(point)
        if any(entry[f'{metric}_{suffix}'] != curves[-1][f'{metric}_{suffix}']
               for metric in ('rmse', 'trr') for suffix in ('mean', 'std')):
            raise ValueError('Curve endpoint differs from table')
        sources.append({'algorithm': method, 'directory': str(directory.resolve()),
                        'metrics_sha256': sha256(directory / 'eval_metrics.csv')})
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'step_statistics.csv', curves)
    write_csv(output / 'per_episode_endpoints.csv', endpoints)
    write_json(output / 'sources.json', {'sources': sources, 'std': 'population, ddof=0'})
    return summary


def run(data, output, maps=MAPS, seeds=tuple(range(1000, 1050))):
    data, output = Path(data).resolve(), validate_output_path(output).resolve()
    origins = json.loads((data / 'kernel_parameters.json').read_text())
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError('Require nonempty unique seeds')
    if not maps or len(set(maps)) != len(maps) or not set(maps) <= set(MAPS):
        raise ValueError('Require unique test2..test5 map names')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'original_artifact_hashes.json'
    if not manifest_path.exists():
        protected = [p for p in (ROOT / 'results').rglob('*') if p.is_file() and output not in p.parents]
        protected += [p for p in (ROOT / 'data').rglob('*') if p.is_file()]
        protected += [p for p in data.rglob('*') if p.is_file()]
        write_json(manifest_path, {str(p.resolve()): sha256(p) for p in protected})
    manifest = json.loads(manifest_path.read_text())
    if any(sha256(path) != expected for path, expected in manifest.items()):
        raise ValueError('Protected original artifacts changed')
    all_summary, directories = [], []
    for scene in maps:
        field = FieldModel.load(data / f'{scene}.pkl')
        runs = {}
        for method in METHODS:
            runs[method] = evaluate_method(method, scene, field, origins[scene]['origin_degrees'], output / scene / method, seeds)
            directories.append(str(runs[method][2].resolve()))
        all_summary.extend(summarize_scene(scene, runs, output / scene))
    if any(sha256(path) != expected for path, expected in manifest.items()):
        raise ValueError('Protected artifacts changed during evaluation')
    write_csv(output / 'summary.csv', all_summary)
    write_json(output / 'validation.json', {'passed': True, 'maps': list(maps), 'methods': list(METHODS),
               'seeds': list(seeds), 'evaluation_groups': len(directories), 'directories': directories,
               'episodes': len(maps)*6*len(seeds), 'metric_rows': len(maps)*6*len(seeds)*72,
               'action_rows': len(maps)*6*len(seeds)*70, 'protected_artifacts_unchanged': len(manifest),
               'paired_starts': True, 'trajectory_measurements_verified': True,
               'checkpoint_weights_unchanged': True, 'curve_endpoints_match_tables': True})
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/comparison/test_maps_seed0')
    parser.add_argument('--maps', nargs='+', choices=MAPS, default=list(MAPS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(1000, 1050)))
    args = parser.parse_args()
    print(run(args.data, args.output, args.maps, args.seeds))
