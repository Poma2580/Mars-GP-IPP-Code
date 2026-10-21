"""Masked sampling/update consistency, three-term rewards and legacy compatibility."""
from copy import deepcopy
from unittest.mock import patch
import numpy as np
import pytest
import torch
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.envs.actions import ACTIONS
from mars_ipp.agents.ppo.masking import action_mask
from mars_ipp.evaluation.evaluator import evaluate
from mars_ipp.experiments.v3_variants import VariantEnv, scaler_for, validate_config
from mars_ipp.experiments.v3_pipeline import load_policy, train_variant
from mars_ipp.experiments.v3_masked import (audit_scene, audit_summary, audit_training,
    check_outputs, directories, plot_training, replot)
from mars_ipp.training.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from mars_ipp.utils.config import ROOT, load_config


def config(version):
    c = load_config(ROOT / f'configs/main/{version}.yaml'); c['ppo']['device'] = 'cpu'
    return c


@pytest.mark.parametrize('position', [(7, 8), (0, 7), (15, 7), (7, 0), (7, 15), (0, 0), (0, 15), (15, 0), (15, 15)])
def test_mask_order_probabilities_and_1000_samples(position):
    mask = action_mask(position)
    expected = [all(0 <= x < 16 for x in np.array(position) + delta) for delta in ACTIONS]
    np.testing.assert_array_equal(mask, expected)
    c = config('v3_3_4'); agent = PPO(c['ppo'], scaler_for(c))
    obs = np.r_[position, np.ones(9)]
    distribution = agent.network.distribution(torch.tensor(agent.prepare_state(obs)), mask)
    assert torch.all(distribution.probs[~torch.tensor(mask)] == 0)
    assert distribution.probs.sum().item() == pytest.approx(1)
    generator = torch.Generator().manual_seed(42)
    for _ in range(1000):
        action, lp, _, _ = agent.act(obs, generator)
        assert mask[action] and np.isfinite(lp)
    assert mask[agent.select_action(obs)]


def test_masked_update_uses_copied_historical_masks_and_finite_gradients():
    c = config('v3_3_4'); c['ppo']['readonly_diagnostics'] = True
    agent = PPO(c['ppo'], scaler_for(c)); expected_masks = []
    for i, position in enumerate([(0, 0), (15, 15), (0, 15), (15, 0)]):
        obs = np.r_[position, np.ones(9)]; mask = agent.mask_for(obs)
        action, logprob, value, state = agent.act(obs)
        lp, _, entropy = agent.network.evaluate(torch.tensor(state), torch.tensor(action), mask)
        assert lp.item() == pytest.approx(logprob)
        p = agent.network.distribution(torch.tensor(state), mask).probs[mask]
        assert entropy.item() == pytest.approx(-(p * p.log()).sum().item())
        expected_masks.append(mask.copy())
        agent.buffer.add(state, action, logprob, value, i * .1, 0., True, True, mask)
        mask[:] = False
    np.testing.assert_array_equal([t[8] for t in agent.buffer.transitions], expected_masks)
    with patch.object(agent, 'mask_for', side_effect=AssertionError('must not regenerate masks')):
        report = agent.update()
    assert all(np.isfinite(v) for v in report.values() if v is not None)
    assert all(torch.isfinite(p.grad).all() for p in agent.network.parameters())
    obs = np.r_[[0, 0], np.ones(9)]
    a, lp, v, state = agent.act(obs)
    agent.buffer.add(state, a, lp, v, 0., 0., True, True)
    with pytest.raises(ValueError, match='historical'):
        agent.update()


@pytest.mark.parametrize('version,dimension', [('v3_3_4', 11), ('v3_5', 14)])
def test_rewards_state_checkpoint_and_evaluator(version, dimension, small_field, tmp_path):
    c = config(version); scaler = scaler_for(c)
    env = VariantEnv(small_field, scaler, c)
    obs, reset = env.reset(options={'start': [2, 2]})
    assert obs.shape == (dimension,)
    cumulative = 0.
    for action in [1, 4, 6]:
        old_b = env.measurement.copy(); revision = env.gp.revision
        _, reward, _, _, info = env.step(action)
        cumulative += info['r_gp']
        components = env.reward_names
        assert reward == pytest.approx(sum(info[f'weighted_r_{k}'] for k in components))
        assert info['trace_prior'] == reset['trace_prior']
        if action == 6:
            assert reward == -.014 and env.gp.revision == revision
            assert info['r_gp'] == 0
            if version == 'v3_5':
                assert info['r_mh'] == 0
        elif version == 'v3_5':
            expected = np.linalg.norm(env.measurement - old_b) / (np.linalg.norm(ACTIONS[action]) * env.bref)
            assert info['r_mh'] == pytest.approx(expected)
            assert info['weighted_r_mh'] == pytest.approx(.02 * expected)
    assert cumulative == pytest.approx((reset['trace_current'] - info['trace_current']) / reset['trace_prior'], abs=1e-7)
    env.reset(options={'start': [0, 0]})
    with pytest.raises(AssertionError, match='outside'):
        env.step(3)
    agent = PPO(c['ppo'], scaler); path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, agent, c, {'completed': True})
    restored, _ = load_policy(path)
    assert restored.action_mask_enabled and restored.input_dim == dimension
    bad = deepcopy(c); bad['ppo']['action_mask'] = False
    with pytest.raises(ValueError, match='action_mask'):
        validate_config(bad)
    with pytest.raises(ValueError, match='mask'):
        restore_checkpoint(read_checkpoint(path), PPO(bad['ppo'], scaler))
    c['evaluation']['seeds'] = [1000, 1001]
    rows, paths, _ = evaluate(small_field, scaler, c, restored)
    audit_scene({f'V{version[1:].replace("_", ".")}': rows}, {f'V{version[1:].replace("_", ".")}': paths}, [1000, 1001])
    assert all(r['outside_actions'] == 0 for r in rows)


def test_zero_field_mh_protection(small_field):
    c = config('v3_5')
    with patch.object(small_field, 'truth', side_effect=lambda p: np.zeros((len(p), 3))), patch.object(small_field, 'measure', return_value=np.zeros(3)):
        env = VariantEnv(small_field, scaler_for(c), c); obs, _ = env.reset(options={'start': [0, 0]})
        assert np.isfinite(obs).all()
        for a in [1, 3]:
            _, reward, _, _, info = env.step(a)
            assert info['r_mh'] == info['weighted_r_mh'] == 0 and np.isfinite(reward)


def test_legacy_sampling_still_exact_and_optional_mask_absent():
    c = config('V3_3_3'); agent = PPO(c['ppo'], scaler_for(c)); obs = np.r_[[0, 0], np.ones(9)]
    expected_generator = torch.Generator().manual_seed(25); generator = torch.Generator().manual_seed(25)
    expected = torch.multinomial(agent.network.actor(torch.tensor(agent.prepare_state(obs))), 1, generator=expected_generator).item()
    assert agent.act(obs, generator)[0] == expected and agent.mask_for(obs) is None


def test_path_guards_and_formal_names(tmp_path):
    train, test = directories(tmp_path / 'train', tmp_path / 'test', False)
    assert train['v3_3_4'].name == 'v_3.3.4_train' and test['v3_5'].name == 'v_3.5_test'
    check_outputs([tmp_path / 'empty']); (tmp_path / 'empty').mkdir(); check_outputs([tmp_path / 'empty'])
    (tmp_path / 'empty/a').write_text('protected')
    with pytest.raises(FileExistsError):
        check_outputs([tmp_path / 'empty'])
    with pytest.raises(ValueError):
        check_outputs([tmp_path, tmp_path / 'child'])


@pytest.mark.parametrize('corruption', [None, 'weighted', 'episode', 'outside'])
def test_training_audit_columns_and_episode_groups(tmp_path, monkeypatch, corruption):
    import json
    from mars_ipp.experiments import v3_masked
    c = load_config(ROOT / 'configs/main/v3_5.yaml')
    c['training'].update(episodes=30, checkpoint_every=10, eval_every=10)
    rows = []; steps = []
    for ep in range(1, 31):
        scene = f'train{(ep - 1) % 5 + 1}'
        rows.append(dict(episode=str(ep), map=scene, reward='0.07', r_gp='0.07', r_rep='0', r_mh='0',
                         weighted_r_gp='0.07', weighted_r_rep='0', weighted_r_mh='0',
                         outside_actions='0', unique_positions='71', repeated_visits='0', raw_trace_sum='0.07', trace_prior='1'))
        for step in range(1, 71):
            steps.append(dict(episode=str(ep), map=scene, step=str(step), reward='0.001', r_gp='0.001', r_rep='0', r_mh='0',
                              weighted_r_gp='0.001', weighted_r_rep='0', weighted_r_mh='0', outside_actions='0'))
    if corruption == 'weighted':
        steps[5]['weighted_r_mh'] = '0.1'
    elif corruption == 'episode':
        steps[5]['episode'] = '999'
    elif corruption == 'outside':
        steps[5]['outside_actions'] = '1'
    monitor = [dict(episode=str(ep), map=f'train{i}', seed=str(seed), outside_actions='0')
               for ep in (10, 20, 30) for i in range(1, 6) for seed in c['training']['monitor_seeds']]
    logs = {'train_metrics.csv': rows, 'train_steps.csv': steps, 'updates.csv': [{'samples': '2100'}], 'eval_endpoints.csv': monitor}
    monkeypatch.setattr(v3_masked, 'read_csv', lambda p: logs[p.name])
    monkeypatch.setattr(v3_masked, 'load_policy', lambda p: (None, {'config': c, 'progress': {'episode':30, 'steps':2100, 'completed':True}}))
    (tmp_path / 'others').mkdir(); (tmp_path / 'figure').mkdir()
    (tmp_path / 'checkpoint.pt').write_bytes(b'fixture')
    (tmp_path / 'others/run_status.json').write_text(json.dumps({'completed':True,'pending_rollout':0}))
    for name in ('training', 'training_evaluation'):
        (tmp_path / f'figure/{name}.png').touch()
    if corruption:
        with pytest.raises((AssertionError, ValueError)):
            v3_masked.audit_training(tmp_path, 'v3_5', True)
    else:
        assert v3_masked.audit_training(tmp_path, 'v3_5', True)['passed']
