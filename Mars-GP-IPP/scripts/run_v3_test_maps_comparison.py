"""Frozen V3/V3.1 seed-0 policies versus local GP-Trace on test2..test5."""
import _bootstrap
import argparse
from copy import deepcopy
import json
import logging
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
import yaml
from threadpoolctl import threadpool_limits
from run_test_maps_comparison import MAPS, DEFAULT_DATA, sha256, validate_rows
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.envs.observations import scaler_from_dict, make_scaler
from mars_ipp.evaluation.evaluator import evaluate
from mars_ipp.field.field_model import FieldModel
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.training.checkpoint import read_checkpoint, restore_checkpoint, scalers_equivalent
from mars_ipp.utils.config import ROOT, merge, validate_output_path
from mars_ipp.utils.seed import seed_all
from mars_ipp.visualization.learning_curves import read_csv, save_figure
from mars_ipp.visualization.comparison import _step_statistics

METHODS = ('v3', 'v3_1', 'gp_trace')
LABELS = {'v3': 'V3', 'v3_1': 'V3.1', 'gp_trace': 'GP-Trace'}
COLORS = {'v3': 'tab:blue', 'v3_1': 'tab:green', 'gp_trace': 'tab:orange'}
CHECKPOINTS = {method: ROOT / 'results/main' / experiment / 'seed_0/checkpoint.pt'
               for method, experiment in (
                   ('v3', 'v3_gp_trace_train1'), ('v3_1', 'v3_1_gp_trace_global_raw_train1'))}
ENV = {'grid_size': 16, 'coordinate_scale': 2., 'evaluation_size': 31,
       'evaluation_extent': 30., 'max_steps': 70, 'noise_rate': 0.,
       'online_noise_variance': 1e-6, 'unique_samples': True}
DEFAULT_OUTPUT = ROOT / 'results/comparison/v3_1_gp_trace_global_raw_train1/test2-test5'
LOG = logging.getLogger('v3_test_maps')


def frozen_agent(method):
    payload = read_checkpoint(CHECKPOINTS[method])
    cfg = payload['config']
    mode, gamma = ('gp_trace_local_raw', .99) if method == 'v3' else ('gp_trace_global_raw', 1.)
    if (cfg['seed'] != 0 or payload['progress'] != {'episode': 10000, 'steps': 700000, 'completed': True}
            or payload.get('input_dim') != 11 or cfg['ppo']['observation_mode'] != 'gp_trace_local_11d'
            or cfg['reward'] != {'mode': mode, 'weights': [1.], 'disabled': []}
            or cfg['ppo']['gamma'] != gamma or cfg['env'] != ENV):
        raise ValueError(f'Require original completed seed-0 checkpoint and protocol: {method}')
    scaler = scaler_from_dict(payload['scaler'])
    fields = prepare_fields(cfg)
    if (fields['train1'].sha256 != cfg['v3']['sha256']
            or not scalers_equivalent(payload['scaler'], make_scaler(fields, cfg).to_dict())):
        raise ValueError('Training data or scaler no longer matches checkpoint')
    agent = PPO(cfg['ppo'], scaler)
    restore_checkpoint(payload, agent)
    return agent, payload


def evaluate_policy(method, scene, field, seeds):
    seed_all(0)
    agent, payload = frozen_agent(method)
    config = merge(payload['config'], {'field': {'test_scenario': scene, 'models': {scene: field.source}},
                                      'evaluation': {'seeds': list(seeds)}})
    scaler = agent.scaler
    scaler_before = deepcopy(scaler.to_dict())
    weights = {key: value.detach().clone() for key, value in agent.network.state_dict().items()}
    LOG.info('Evaluate %s/%s: %d episodes; checkpoint=%s; sha256=%s; scaler=%s',
             scene, LABELS[method], len(seeds), CHECKPOINTS[method], sha256(CHECKPOINTS[method]), scaler_before)
    rows, paths, _ = evaluate(field, scaler, config, agent, 'ppo', episode=10000)
    if (agent.scaler is not scaler or agent.scaler.to_dict() != scaler_before
            or any(not value.equal(agent.network.state_dict()[key]) for key, value in weights.items())):
        raise ValueError('Evaluation changed checkpoint weights or original scaler')
    for row in (*rows, *paths):
        row['algorithm'] = method
    validate_rows(rows, paths, seeds, method, 10000, field)
    if method == 'v3_1':
        errors = []
        for seed in seeds:
            selected = [r for r in rows if int(r['seed']) == seed]
            reset, final = selected[1], selected[-1]
            total = sum(float(r['r_gp']) for r in selected[2:])
            delta = float(reset['trace_current']) - float(final['trace_current'])
            if (not np.isfinite([total, delta]).all()
                    or not np.isclose(total, delta, rtol=2e-7, atol=1e-5)
                    or not np.isclose(total, final['sum_reward'], rtol=2e-7, atol=1e-5)
                    or final['trace_reset'] != reset['trace_current']):
                raise ValueError('V3.1 rewards do not telescope from reset Trace')
            errors.append(abs(total - delta))
        LOG.info('%s V3.1 max_abs_trace_identity_error=%.12g', scene, max(errors))
    return rows


def gp_trace_rows(scene, field, seeds):
    """Reuse only verified old metric/action rows; no writes to their source."""
    source = ROOT / 'results/comparison/test_maps_seed0' / scene / 'gp_trace'
    try:
        cfg = yaml.safe_load((source / 'config.yaml').read_text())
        meta = json.loads((source / 'metadata.json').read_text())
        if (cfg['env'] != ENV or cfg['field']['test_scenario'] != scene
                or cfg['evaluation']['seeds'] != list(seeds)
                or meta['algorithm'] != 'gp_trace' or meta['checkpoint'] is not None
                or meta['field']['sha256'] != field.sha256
                or meta['field']['parameters'] != field.parameters
                or meta['gp_protocol'] != {'noise_variance': 1e-6, 'unique_samples': True, 'known_map_kernel': True}):
            raise ValueError('GP-Trace source protocol or field differs')
        for name in ('src/mars_ipp/agents/baselines/gp_trace.py',
                     'src/mars_ipp/agents/baselines/gp_greedy.py',
                     'src/mars_ipp/gp/uncertainty.py', 'src/mars_ipp/gp/estimator.py',
                     'src/mars_ipp/envs/actions.py'):
            if meta['source_sha256'][name] != sha256(ROOT / name):
                raise ValueError(f'GP-Trace implementation differs: {name}')
        rows = read_csv(source / 'eval_metrics.csv')
        paths = read_csv(source / 'trajectories.csv')
        validate_rows(rows, paths, seeds, 'gp_trace', 0, field)
        LOG.info('Reuse verified %s/GP-Trace: source=%s; metrics_sha256=%s; trajectories_sha256=%s',
                 scene, source, sha256(source / 'eval_metrics.csv'), sha256(source / 'trajectories.csv'))
        return rows
    except (OSError, ValueError, KeyError, TypeError) as exc:
        LOG.info('Reevaluate %s/GP-Trace (local 8-neighbor greedy): %s', scene, exc)
    # Baseline ignores the observation vector; sharing the fixed 11D protocol is safe.
    _, payload = frozen_agent('v3')
    cfg = merge(payload['config'], {'field': {'test_scenario': scene, 'models': {scene: field.source}},
                                   'evaluation': {'seeds': list(seeds)}})
    seed_all(0)
    rows, paths, _ = evaluate(field, scaler_from_dict(payload['scaler']), cfg, algorithm='gp_trace')
    validate_rows(rows, paths, seeds, 'gp_trace', 0, field)
    return rows


def paired_statistics(runs, seeds):
    reference = None
    summary = {}
    for method in METHODS:
        rows = runs[method]
        initial = {int(r['seed']): np.array([float(r[k]) for k in
                   ('start_i', 'start_j', 'trace_prior', 'trace_current', 'rmse')])
                   for r in rows if r['stage'] == 'reset'}
        if set(initial) != set(seeds):
            raise ValueError('Missing paired reset rows')
        if reference is not None:
            for seed in seeds:
                np.testing.assert_allclose(initial[seed], reference[seed], rtol=1e-10, atol=1e-8)
        reference = initial
        final = [r for r in rows if r['stage'] == 'step' and int(r['step']) == 70]
        if sorted(int(r['seed']) for r in final) != sorted(seeds):
            raise ValueError('Missing or duplicate endpoints')
        summary[method] = {}
        for metric in ('rmse', 'trr'):
            values = np.array([float(r[metric]) for r in final])
            steps, mean, std = _step_statistics(rows, method, metric)
            if steps != list(range(71)) or not np.isfinite([mean, std]).all():
                raise ValueError('Incomplete or nonfinite plot curve')
            endpoint = (float(values.mean()), float(values.std(ddof=0)))
            np.testing.assert_allclose([mean[-1], std[-1]], endpoint, rtol=1e-12, atol=1e-12)
            summary[method][metric] = endpoint
    return summary


def save_scene(scene, runs, seeds, output):
    summary = paired_statistics(runs, seeds)
    files = (output / 'figures/rmse.png', output / 'figures/trr.png', output / 'comparison.tex')
    if any(path.exists() for path in files):
        raise FileExistsError(f'Refuse to overwrite existing scene outputs: {output}')
    for metric, label in (('rmse', 'RMSE (nT)'), ('trr', 'Trr (%)')):
        fig, ax = plt.subplots(figsize=(9, 5.625))
        for method in METHODS:
            steps, mean, std = _step_statistics(runs[method], method, metric)
            ax.plot(steps, mean, color=COLORS[method], linewidth=2,
                    label=f'{LABELS[method]} (n={len(seeds)})')
            ax.fill_between(steps, mean - std, mean + std, color=COLORS[method], alpha=.15)
        ax.set(xlabel='Step', ylabel=label, title=scene, xlim=(0, 70))
        ax.grid(alpha=.35)
        ax.legend(loc='upper right')
        save_figure(fig, output / 'figures' / metric, ('png',))
    lines = [r'\begin{tabular}{lcc}', r'\hline',
             r'Method & RMSE (nT) & Trr (\%) \\', r'\hline']
    for method in METHODS:
        cells = [f'${summary[method][metric][0]:.4f} \\pm {summary[method][metric][1]:.4f}$'
                 for metric in ('rmse', 'trr')]
        lines.append(LABELS[method] + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\hline', r'\end{tabular}']
    # Exclusive creation also protects against concurrent writes.
    with (output / 'comparison.tex').open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    LOG.info('%s endpoint_mean_std_ddof0=%s', scene, json.dumps(summary))
    return summary


def run(data, output, maps=MAPS, seeds=tuple(range(1000, 1050))):
    data, output = Path(data).resolve(), validate_output_path(output).resolve()
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Require nonempty unique test seeds')
    if not maps or len(set(maps)) != len(maps) or not set(maps) <= set(MAPS):
        raise ValueError('Require unique maps from test2..test5')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Result directory is not empty: {output}')
    inputs = [*CHECKPOINTS.values(), *(data / f'{scene}.pkl' for scene in maps)]
    inputs += [p for scene in maps for p in
               (ROOT / 'results/comparison/test_maps_seed0' / scene / 'gp_trace').rglob('*') if p.is_file()]
    hashes = {str(path): sha256(path) for path in inputs}
    LOG.info('Start maps=%s seeds=%s output=%s input_sha256=%s', maps, seeds, output, json.dumps(hashes))
    LOG.info('Protocol: frozen seed-0; known per-map fixed GP kernel; 70 actions; 961 latent variance points; '
             'noise=1e-6; unique samples; categorical sampling; original 11D scaler; ddof=0')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with threadpool_limits(limits=1):
        for scene in maps:
            field = FieldModel.load(data / f'{scene}.pkl')
            LOG.info('Map %s sha256=%s parameters=%s', scene, field.sha256, field.parameters)
            runs = {method: evaluate_policy(method, scene, field, seeds) for method in ('v3', 'v3_1')}
            runs['gp_trace'] = gp_trace_rows(scene, field, seeds)
            save_scene(scene, runs, seeds, output / scene)
            LOG.info('Completed %s: paired starts/initial GP, weights/scalers, curves/table verified', scene)
    for name, digest in hashes.items():
        if sha256(name) != digest:
            raise ValueError(f'Input artifact changed: {name}')
    expected = {output / scene / relative for scene in maps
                for relative in ('figures/rmse.png', 'figures/trr.png', 'comparison.tex')}
    actual = {p for p in output.rglob('*') if p.is_file()}
    if actual != expected:
        raise ValueError('Unexpected output file set')
    LOG.info('SUCCESS: %d PNG + %d tex; %d input files unchanged; elapsed_seconds=%.3f',
             2 * len(maps), len(maps), len(hashes), time.monotonic() - started)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--maps', nargs='+', choices=MAPS, default=list(MAPS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(1000, 1050)))
    parser.add_argument('--log', type=Path, default=ROOT / 'logs/v3_test2_test5.log')
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(args.log), logging.StreamHandler()])
    LOG.setLevel(logging.INFO)
    try:
        run(args.data, args.output, tuple(args.maps), tuple(args.seeds))
    except Exception:
        LOG.exception('Evaluation failed; no training was started')
        raise


if __name__ == '__main__':
    main()
