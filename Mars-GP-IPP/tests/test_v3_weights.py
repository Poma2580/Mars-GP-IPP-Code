"""Reward coefficients, legacy compatibility and isolated paired artifacts."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json

import numpy as np
import pytest
import torch

from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.envs.actions import ACTIONS
from mars_ipp.evaluation.evaluator import evaluate
from mars_ipp.experiments.v3_variants import WEIGHT_EXPERIMENTS, VariantEnv, scaler_for, validate_config
from mars_ipp.experiments.v3_pipeline import scene_report, train_variant, load_policy
from mars_ipp.experiments.v3_weights import (audit_scene, audit_training, check_new, group_root,
                                           names, preflight, require_formal, replot, audit_report)
from mars_ipp.gp.uncertainty import trace
from mars_ipp.training.checkpoint import read_checkpoint, save_checkpoint
from mars_ipp.utils.config import ROOT, load_config
from mars_ipp.utils.logger import write_csv, write_json


def config(name):
    c = load_config(ROOT / f'configs/main/{name}.yaml')
    c['ppo']['device'] = 'cpu'
    return c


def action(delta):
    return next(i for i, d in enumerate(ACTIONS) if list(d) == delta)


@pytest.mark.parametrize('name', WEIGHT_EXPERIMENTS)
def test_weight_config_reward_and_checkpoint(name, small_field, tmp_path):
    c = config(name); version, weight = WEIGHT_EXPERIMENTS[name]
    base = config(version); expected = deepcopy(base)
    expected['experiment'] = name; expected['reward']['weights'] = [1., weight]
    assert c == expected
    env = VariantEnv(small_field, scaler_for(c), c)
    obs, info = env.reset(options={'start': [2, 2]})
    assert obs.shape == (11,)
    prior = info['trace_prior']
    for delta in ([1, 0], [1, 1]):
        before = trace(env.gp.predict(env.evaluation_points)[1])
        old_b = env.measurement.copy()
        _, reward, _, _, info = env.step(action(delta))
        gp = (before - trace(env.gp.predict(env.evaluation_points)[1])) / prior
        mh = np.linalg.norm(env.measurement - old_b) / (np.linalg.norm(delta) * env.bref)
        assert reward == pytest.approx(gp + (weight * mh if version == 'v3_4' else 0), abs=1e-7)
        assert reward == pytest.approx(info['reward_total'])
        assert info['trace_prior'] == prior
    revision = env.gp.revision
    with patch.object(small_field, 'measure', side_effect=AssertionError('repeat measurement')):
        _, reward, _, _, info = env.step(action([-1, -1]))
    assert reward == (-weight if version == 'v3_3' else 0)
    assert env.gp.revision == revision and not info['sampled']
    env.reset(options={'start': [0, 0]}); revision = env.gp.revision
    with patch.object(small_field, 'measure', side_effect=AssertionError('outside measurement')):
        _, reward, _, _, info = env.step(action([-1, 0]))
    assert reward == 0 and info['reward_total'] == 0 and env.gp.revision == revision
    agent = PPO(c['ppo'], scaler_for(c)); path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, agent, c, {'completed': False})
    other, payload = load_policy(path)
    assert payload['config'] == c and other.input_dim == 11
    assert all(torch.equal(a, b) for a, b in zip(agent.network.parameters(), other.network.parameters()))
    bad = deepcopy(c); bad['reward']['weights'][1] = -weight
    with pytest.raises(ValueError):
        validate_config(bad)
    bad['reward']['weights'] = [.5, .5]
    with pytest.raises(ValueError):
        validate_config(bad)


def test_legacy_weight_and_unknown_experiment_rejected():
    c = config('v3_3'); c['reward']['weights'] = [1., .005]
    with pytest.raises(ValueError):
        validate_config(c)
    c['experiment'] = 'V3_3_99'
    with pytest.raises(ValueError, match='Unknown'):
        validate_config(c)
    c = config('V3_3_3'); c['version'] = 'v3_4'; c['ppo']['variant'] = 'v3_4'
    with pytest.raises(ValueError, match='base version'):
        validate_config(c)


@pytest.mark.parametrize('name', names('v3_4'))
def test_zero_field_mh(name, small_field):
    c = config(name)
    with patch.object(small_field, 'truth', side_effect=lambda p: np.zeros((len(p), 3))), \
         patch.object(small_field, 'measure', return_value=np.zeros(3)):
        env = VariantEnv(small_field, scaler_for(c), c); env.reset(options={'start': [0, 0]})
        for delta in ([1, 0], [-1, 0], [-1, 0]):
            _, reward, _, _, info = env.step(action(delta))
            assert info['r_mh'] == info['weighted_r_mh'] == 0
            assert np.isfinite(reward)


def test_full_rollout_identity_and_replay(small_field):
    name = 'V3_3_3'; c = config(name); c['evaluation']['seeds'] = [1000, 1001]
    agent = PPO(c['ppo'], scaler_for(c))
    rows, paths, _ = evaluate(small_field, agent.scaler, c, agent)
    audit_scene({name: rows}, {name: paths}, [1000, 1001], [name], {name: c})
    bad = deepcopy(paths); bad[3]['i'] += 1
    with pytest.raises(ValueError, match='position mismatch'):
        audit_scene({name: rows}, {name: bad}, [1000, 1001], [name], {name: c})
    for seed in (1000, 1001):
        selected = [r for r in rows if r['seed'] == seed and r['stage'] == 'step']
        reset = next(r for r in rows if r['seed'] == seed and r['stage'] == 'reset')
        assert sum(r['r_gp'] for r in selected) == pytest.approx(
            (reset['trace_current'] - selected[-1]['trace_current']) / reset['trace_prior'], abs=1e-7)


def test_dynamic_report_and_csv_only_replot(small_field, tmp_path, monkeypatch):
    import mars_ipp.experiments.v3_weights as module
    c = config('V3_3_1'); c['evaluation']['seeds'] = [1000, 1001]
    agent = PPO(c['ppo'], scaler_for(c)); rows, paths, _ = evaluate(small_field, agent.scaler, c, agent)
    for row in rows:
        row['rmse'] = 2. if row['seed'] == 1000 else 4.
    for group in ('v3_3', 'v3_4'):
        methods = ('gp_trace', *names(group))
        root = group_root(tmp_path / 'test', group, True)
        write_json(root / 'others/metadata.json', {'smoke': True, 'seeds': [1000, 1001]})
        for i in range(1, 6):
            scene = f'test{i}'; runs = {m: deepcopy(rows) for m in methods}
            trajectories = {m: deepcopy(paths) for m in methods}
            for method in methods:
                base = root / f'others/{scene}/{method}'
                write_csv(base / 'eval_metrics.csv', rows); write_csv(base / 'trajectories.csv', paths)
            summary = scene_report(scene, runs, trajectories, root / scene, [1000, 1001], True,
                                   methods=methods, note=module.NOTE, aux_output=root / f'others/{scene}', weight_report=True)
            assert len(summary) == 3 * len(methods)
            assert all(r['rmse_mean'] == 3 and r['rmse_std'] == 1 for r in summary)
            table = (root / f'{scene}/table/comparison.md').read_text()
            assert '| 方法 | RMSE (nT) | 最终 Trace (nT²) | Trr (%) | 唯一点 | 重复 | 越界 |' in table
            assert all(f'预算{budget}步表现' in table for budget in (30, 50, 70))
            audit_report(root, scene, runs, methods, [1000, 1001])
            if group == 'v3_3' and i == 1:
                table_path = root / f'{scene}/table/comparison.md'
                table_path.write_text(table.replace('3.0000 ± 1.0000', '3.0000 ± 2.0000', 1))
                with pytest.raises(ValueError, match='Markdown'):
                    audit_report(root, scene, runs, methods, [1000, 1001])
                table_path.write_text(table)
    monkeypatch.setattr(module, 'plot_training', lambda *_: None)
    monkeypatch.setattr(module, 'load_policy', lambda *_: pytest.fail('Replot loaded a policy'))
    monkeypatch.setattr(module.FieldModel, 'load', lambda *_: pytest.fail('Replot loaded field'))
    replot(tmp_path / 'train', tmp_path / 'test', tmp_path / 'replot', True)
    assert json.loads((tmp_path / 'replot/acceptance.json').read_text())['passed']
    with pytest.raises(FileExistsError):
        replot(tmp_path / 'train', tmp_path / 'test', tmp_path / 'replot', True)


def test_smoke_training_audit_and_formal_guard(small_field, tmp_path, monkeypatch):
    import mars_ipp.experiments.v3_pipeline as pipeline
    from mars_ipp.experiments.v3_weights import plot_training
    c = config('V3_3_3'); c['training']['monitor_seeds'] = [1000]
    monkeypatch.setattr(pipeline.FieldModel, 'load', lambda _: small_field)
    directory = tmp_path / 'smoke'
    train_variant(c, directory, True, plotter=plot_training)
    result = audit_training(directory, 'V3_3_3', True)
    assert result['update_samples'] == [2000, 100]
    assert result['map_counts'] == dict.fromkeys([f'train{i}' for i in range(1, 6)], 6)
    with pytest.raises(ValueError, match='non-smoke'):
        require_formal(directory, 'V3_3_3')
    with pytest.raises(FileExistsError):
        train_variant(c, directory, True)


def test_preflight_no_output_and_conflict(tmp_path, monkeypatch):
    import mars_ipp.experiments.v3_weights as module
    monkeypatch.setattr(module.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(module.torch.cuda, 'get_device_name', lambda: 'test GPU')
    report = preflight(tmp_path / 'train', tmp_path / 'test', True)
    assert len(report['configs']) == 7 and not list(tmp_path.iterdir())
    (tmp_path / 'exists').mkdir()
    with pytest.raises(FileExistsError):
        check_new([tmp_path / 'exists'])
    with pytest.raises(ValueError):
        check_new([tmp_path / 'new', tmp_path / 'new/sub'])


def test_smoke_audit_failure_preserves_record_before_evaluation(tmp_path, monkeypatch):
    import mars_ipp.experiments.v3_weights as module
    c = config('V3_3_1')
    monkeypatch.setattr(module, 'preflight', lambda *_: {'configs': {'V3_3_1': c}, 'data': {}})
    monkeypatch.setattr(module, 'prepare_data', lambda **_: {})
    monkeypatch.setattr(module, 'protect_history', lambda: {})
    monkeypatch.setattr(module, 'verify_hashes', lambda _: None)
    monkeypatch.setattr(module, 'WEIGHT_EXPERIMENTS', {'V3_3_1': ('v3_3', .001)})
    def train(config, output, *args, **kwargs):
        output.mkdir(parents=True)
    def fail(*_):
        raise ValueError('audit rejected result')
    monkeypatch.setattr(module, 'train_variant', train)
    monkeypatch.setattr(module, 'audit_training', fail)
    with pytest.raises(ValueError, match='audit rejected'):
        module.run('smoke', tmp_path / 'train', tmp_path / 'test')
    path = group_root(tmp_path / 'train', 'v3_3', True) / 'others/failure.json'
    assert json.loads(path.read_text())['stage'] == 'smoke'
    assert not (tmp_path / 'test').exists()
