from copy import deepcopy
from unittest.mock import patch
import importlib
import json
import numpy as np
import pytest
import torch
import yaml

from mars_ipp.utils.config import ROOT, load_config
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.envs.observations import make_scaler
from mars_ipp.envs.actions import legal_neighbors
from mars_ipp.evaluation.evaluator import env_config, evaluate
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.gp.uncertainty import expected_trace_reduction, trace
from mars_ipp.agents.ppo.buffer import compute_gae
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.training.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint
from mars_ipp.visualization.learning_curves import read_csv


def cfg():
    c = load_config(ROOT / 'configs/main/v3_1_gp_trace_global_raw_train1.yaml')
    c['ppo']['device'] = 'cpu'
    return c


def environment(field, mode='terminal'):
    c = cfg()
    return GridEnv(field, make_scaler({'train1': field}, c), env_config(c), metrics_mode=mode)


def test_only_two_algorithm_config_differences():
    v3 = yaml.safe_load((ROOT / 'results/main/v3_gp_trace_train1/seed_0/config.yaml').read_text())
    raw = load_config(ROOT / 'configs/main/v3_1_gp_trace_global_raw_train1.yaml')
    assert raw['reward']['mode'] == 'gp_trace_global_raw' and raw['ppo']['gamma'] == 1.
    raw['experiment'] = v3['experiment']
    raw['reward']['mode'] = v3['reward']['mode']
    raw['ppo']['gamma'] = v3['ppo']['gamma']
    assert raw == v3 == load_config(ROOT / 'configs/main/v3_gp_trace_train1.yaml')


@pytest.mark.parametrize('start', [[0, 0], [8, 8], [15, 15]])
def test_analytic_selected_only_before_sampling(small_field, start):
    env = environment(small_field)
    obs, info = env.reset(options={'start': start})
    assert obs.shape == (11,) and info['trace_reset'] < info['trace_prior']
    for _ in range(5):
        actions, positions = legal_neighbors(env.position, 16)
        action = next(int(a) for a, p in zip(actions, positions) if tuple(p) not in env.visited)
        selected = positions[np.flatnonzero(actions == action)] * 2
        before = trace(env.gp.predict(env.evaluation_points)[1])
        revision = env.gp.revision
        def score(gp, candidates, points):
            assert gp.revision == revision
            np.testing.assert_array_equal(candidates, selected)
            np.testing.assert_array_equal(points, env.evaluation_points)
            assert candidates.shape == (1, 2) and points.shape == (961, 2)
            return expected_trace_reduction(gp, candidates, points)
        with patch('mars_ipp.envs.grid_env.expected_trace_reduction', side_effect=score) as scoring:
            _, reward, _, _, info = env.step(action)
        scoring.assert_called_once()
        after = trace(env.gp.predict(env.evaluation_points)[1])
        assert reward == pytest.approx(before - after, rel=2e-7, abs=1e-5)
        assert reward == info['r_gp'] == info['raw_trace_reward']
        assert info['sampled'] and env.gp.revision == revision + 1


def test_repeat_outside_zero_no_score_no_truth_70_step_budget(small_field):
    env = environment(small_field)
    env.reset(options={'start': [0, 0]})
    env.step(0)
    x, y = env.gp.X.copy(), env.gp.Y.copy()
    revision = env.gp.revision
    before = trace(env.gp.predict(env.evaluation_points)[1])
    with (patch('mars_ipp.envs.grid_env.expected_trace_reduction', side_effect=AssertionError('no score')),
          patch.object(small_field, 'measure', side_effect=AssertionError('no measure'))):
        for _ in range(69):
            _, reward, done, truncated, info = env.step(2)
            assert reward == 0 and not truncated and not info['sampled']
    assert done and env.count == 70 and env.gp.revision == revision
    assert info['repeated_visits'] == 1 and info['outside_actions'] == 68
    np.testing.assert_array_equal(env.gp.X, x)
    np.testing.assert_array_equal(env.gp.Y, y)
    assert info['trace_current'] == before
    assert info['sum_reward'] == pytest.approx(env.trace_reset - before, rel=2e-7, abs=1e-5)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, -1.])
def test_invalid_score_stops_before_sampling(small_field, value):
    env = environment(small_field)
    env.reset(options={'start': [0, 0]})
    with patch('mars_ipp.envs.grid_env.expected_trace_reduction', return_value=np.array([value])), pytest.raises(FloatingPointError):
        env.step(0)
    assert len(env.gp.X) == 1


def test_roundoff_negative_and_positive_not_clipped(small_field):
    env = environment(small_field)
    env.reset(options={'start': [0, 0]})
    tiny = -0.5e-12 * max(1., env.trace_prior)
    with patch('mars_ipp.envs.grid_env.expected_trace_reduction', return_value=np.array([tiny])):
        assert env.step(0)[1] == 0
    with patch('mars_ipp.envs.grid_env.expected_trace_reduction', return_value=np.array([1234.])):
        assert env.step(0)[1] == 1234.


@pytest.mark.parametrize('seed', [1000, 1001, 1002])
def test_real_train1_episode_telescope_and_full_schema(seed):
    fields = prepare_fields(cfg())
    env = environment(fields['train1'], 'full')
    _, reset = env.reset(seed=seed)
    assert env.trace_reset == reset['trace_current'] < env.trace_prior
    schema = set(env.prior_info)
    assert set(reset) == schema
    rng = np.random.default_rng(seed)
    total = 0.
    for _ in range(70):
        _, reward, _, _, info = env.step(int(rng.integers(8)))
        total += reward
        assert set(info) == schema
    assert total == pytest.approx(reset['trace_current'] - info['trace_current'], rel=2e-7, abs=1e-5)
    assert not np.isclose(total, env.trace_prior - info['trace_current'], rtol=2e-7, atol=1e-5)


def test_identity_mismatch_fails(small_field):
    env = environment(small_field)
    env.reset(options={'start': [0, 0]})
    env.max_steps = 1
    with patch('mars_ipp.envs.grid_env.expected_trace_reduction', return_value=np.array([1e6])), pytest.raises(FloatingPointError, match='accumulation'):
        env.step(0)


def test_gamma_one_gae_terminal_truncation_and_rollout():
    adv, targets = compute_gae([1000, 2000, 3000], [10, 20, 30], [20, 99999, 40],
                               [False, True, False], [False, True, False], 1., .9)
    np.testing.assert_allclose(adv, [1010 + .9 * 1980, 1980, 3010])
    np.testing.assert_allclose(targets, adv + [10, 20, 30])
    adv, _ = compute_gae([1, 99], [0, 0], [2, 0], [False, True], [True, True], 1., .9)
    assert adv[0] == 3
    assert np.isfinite(targets).all()


def test_checkpoint_roundtrip_and_gamma_rejection(small_field, tmp_path):
    c = cfg()
    env = environment(small_field)
    obs, _ = env.reset(seed=0)
    agent = PPO(c['ppo'], env.scaler)
    action, lp, value, state = agent.act(obs)
    _, reward, _, _, _ = env.step(action)
    agent.buffer.add(state, action, lp, value, reward, 0., True, True)
    assert np.isfinite(agent.update()['value_loss'])
    path = tmp_path / 'raw.pt'
    save_checkpoint(path, agent, c, {'completed': True, 'episode': 40})
    payload = read_checkpoint(path)
    assert payload['model_version'] == 'ppo-11d-v3'
    restored = PPO(c['ppo'], env.scaler)
    restore_checkpoint(payload, restored)
    for a, b in zip(agent.network.parameters(), restored.network.parameters()):
        assert torch.equal(a, b)
    bad = deepcopy(payload)
    bad['config']['ppo']['gamma'] = .99
    with pytest.raises(ValueError, match='gamma'):
        restore_checkpoint(bad, restored)
    path = tmp_path / 'bad.yaml'
    path.write_text(f'extends: {ROOT}/configs/main/v3_1_gp_trace_global_raw_train1.yaml\nppo:\n  gamma: 0.99\n')
    with pytest.raises(ValueError, match='gamma'):
        load_config(path)


def test_trainer_logs_raw_steps_and_diagnostics(small_field, tmp_path, monkeypatch):
    import mars_ipp.training.trainer as trainer
    c = cfg()
    c['training'].update(episodes=2, eval_every=2, checkpoint_every=2)
    c['env']['max_steps'] = 3
    monkeypatch.setattr(trainer, 'prepare_fields', lambda _: {'train1': small_field})
    directory = trainer.train(c, tmp_path / 'raw')
    episodes = read_csv(directory / 'train_metrics.csv')
    steps = read_csv(directory / 'raw_trace_steps.csv')
    assert len(episodes) == 2 and len(steps) == 6
    for row in episodes:
        assert float(row['sum_reward']) == pytest.approx(float(row['trace_reset']) - float(row['trace_final']), rel=2e-7, abs=1e-5)
        assert float(row['r_gp']) == float(row['reward'])
    diag = json.loads((directory / 'diagnostics.json').read_text())
    assert diag['trace_identity_checked_episodes'] == 2 and diag['parameters_finite']
    assert diag['raw_rewards']['count'] == 6
    assert read_checkpoint(directory / 'checkpoint.pt')['progress']['completed']


def test_evaluator_endpoints_have_reward_accounting(small_field):
    c = cfg()
    c['evaluation']['seeds'] = [1000]
    c['env']['max_steps'] = 3
    scaler = make_scaler({'train1': small_field}, c)
    rows, _, _ = evaluate(small_field, scaler, c, PPO(c['ppo'], scaler), metrics_mode='terminal')
    assert len(rows) == 1 and rows[0]['trace_reset'] is not None
    assert rows[0]['sum_reward'] == pytest.approx(rows[0]['trace_reset'] - rows[0]['trace_current'], rel=2e-7, abs=1e-5)


def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    return importlib.import_module('run_v3_1_global_raw')


def test_preflight_rejects_extra_config_change(monkeypatch):
    cli = runner(monkeypatch)
    c = load_config(cli.CONFIG)
    c['ppo']['actor_lr'] = .01
    with pytest.raises(ValueError, match='two allowed'):
        cli.preflight(c)


def test_smoke_gate_rejects_changed_source_or_seed(tmp_path, monkeypatch):
    cli = runner(monkeypatch)
    report = {'completed': True, 'training_seeds': [0],
              'config_sha256': cli.sha256(cli.CONFIG), 'source_sha256': cli.source_hashes()}
    (tmp_path / 'smoke_report.json').write_text(json.dumps(report))
    cli.require_smoke(tmp_path, [0])
    with pytest.raises(ValueError):
        cli.require_smoke(tmp_path, [1])
    report['source_sha256'] = {}
    (tmp_path / 'smoke_report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        cli.require_smoke(tmp_path, [0])


def test_reuse_real_historical_results_only_writes_new_copies(tmp_path, monkeypatch):
    cli = runner(monkeypatch)
    import mars_ipp.visualization.comparison as plots
    calls = []
    monkeypatch.setattr(plots, 'plot_results', lambda path, **kwargs: calls.append(path))
    c = load_config(cli.CONFIG)
    for method, source in [('v3_seed_0', 'ppo_seed_0'), ('gp_trace', 'gp_trace')]:
        old = cli.V3_COMPARISON / source
        files = {p.name: cli.sha256(p) for p in old.iterdir() if p.is_file()}
        cli.reuse_historical(c, tmp_path, method)
        copied = tmp_path / method
        assert calls[-1] == copied and not copied.is_symlink()
        assert json.loads((copied / 'reuse.json').read_text())['read_only_source']
        assert cli.sha256(copied / 'eval_metrics.csv') == files['eval_metrics.csv']
        assert all(cli.sha256(old / name) == digest for name, digest in files.items())


def test_history_unrelated_report_audited_without_rewriting_snapshot(tmp_path, monkeypatch):
    cli = runner(monkeypatch)
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'V3_COMPARISON', tmp_path / 'results/comparison/v3_gp_trace_train1')
    output = tmp_path / 'results/comparison/new'
    output.mkdir(parents=True)
    report = tmp_path / 'results/comparison/other/report.md'
    report.parent.mkdir()
    report.write_text('original report')
    manifest = {str(report.relative_to(tmp_path)): cli.sha256(report)}
    snapshot = output / 'historical_hashes.json'
    snapshot.write_text(json.dumps(manifest))
    original = snapshot.read_bytes()
    report.write_text('editorial update')
    assert cli.verify_history(output) == 0
    audit = json.loads((output / 'history_audit.json').read_text())
    assert audit['critical_checks_passed'] and audit['changes'][0]['classification'] == 'unrelated_editorial_report'
    assert snapshot.read_bytes() == original
    assert report.read_text() == 'editorial update'


@pytest.mark.parametrize('name', ['results/comparison/v3_gp_trace_train1/report.md',
    'results/comparison/other/eval_metrics.csv', 'results/main/other/checkpoint.pt',
    'configs/main/other.yaml', 'results/comparison/other/missing/report.md'])
def test_history_critical_changes_and_missing_reports_still_fail(tmp_path, monkeypatch, name):
    cli = runner(monkeypatch)
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'V3_COMPARISON', tmp_path / 'results/comparison/v3_gp_trace_train1')
    output = tmp_path / 'results/comparison/new'
    output.mkdir(parents=True)
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_text('original')
    snapshot = output / 'historical_hashes.json'
    snapshot.write_text(json.dumps({name: cli.sha256(path)}))
    if 'missing' in name:
        path.unlink()
    else:
        path.write_text('changed')
    with pytest.raises(ValueError, match='Historical artifact changed'):
        cli.verify_history(output)
    assert not json.loads((output / 'history_audit.json').read_text())['critical_checks_passed']
