from copy import deepcopy
import numpy as np
import pytest
import torch
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.evaluation.evaluator import env_config, evaluate
from mars_ipp.training.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint
from mars_ipp.utils.config import ROOT, load_config, merge


def test_config_changes_only_policy_mode_and_run_identity():
    full = load_config(ROOT / 'configs/main/original.yaml')
    no_b = load_config(ROOT / 'configs/ablation/no_b_rbf.yaml')
    assert no_b == merge(full, {'experiment': 'no_b_rbf', 'seed': 0, 'seeds': [0],
                              'ppo': {'observation_mode': 'position_only'}})


def test_mask_after_normalization_and_full_compatibility(config, scaler):
    raw = np.array([1., 2., 101., -22., 33.])
    other = np.array([1., 2., -999., 88., -123.])
    original = raw.copy()
    full = PPO(config['ppo'], scaler)
    explicit_full = PPO({**config['ppo'], 'observation_mode': 'full'}, scaler)
    no_b = PPO({**config['ppo'], 'observation_mode': 'position_only'}, scaler)
    np.testing.assert_array_equal(full.prepare_state(raw), scaler.transform(raw))
    np.testing.assert_array_equal(explicit_full.prepare_state(raw), full.prepare_state(raw))
    np.testing.assert_array_equal(no_b.prepare_state(raw), no_b.prepare_state(other))
    np.testing.assert_array_equal(no_b.prepare_state(raw)[2:], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(no_b.prepare_state(raw)[:2], full.prepare_state(raw)[:2])
    np.testing.assert_array_equal(raw, original)
    assert no_b.prepare_state(raw).shape == (5,)
    assert np.any(scaler.transform(np.array([1., 2., 0., 0., 0.]))[2:] != 0)
    with pytest.raises(ValueError, match='observation_mode'):
        PPO({**config['ppo'], 'observation_mode': 'typo'}, scaler)


def test_act_value_buffer_and_update_share_masked_state(config, scaler):
    agent = PPO({**config['ppo'], 'observation_mode': 'position_only'}, scaler)
    observed = {'actor': [], 'critic': []}
    handles = []
    for name in observed:
        handles.append(getattr(agent.network, name).register_forward_pre_hook(
            lambda module, args, name=name: observed[name].append(args[0].detach().cpu().numpy().copy())))
    raw = np.array([1., 2., 101., -22., 33.])
    action, logprob, value, state = agent.act(raw)
    assert agent.value(raw) == value
    np.testing.assert_array_equal(observed['actor'][0], state)
    for critic_state in observed['critic']:
        np.testing.assert_array_equal(critic_state, state)
    agent.buffer.add(state, action, logprob, value, 1., agent.value(raw), False, True)
    np.testing.assert_array_equal(agent.buffer.transitions[0][0][2:], np.zeros(3))
    before = [p.clone() for p in agent.network.parameters()]
    assert np.isfinite(agent.update()['loss'])
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.network.parameters()))
    for states in observed.values():
        for network_state in states:
            assert np.all(network_state[..., 2:] == 0)
    for handle in handles:
        handle.remove()


def test_checkpoint_mode_validation_and_legacy_full(config, scaler, tmp_path):
    no_b_config = merge(config, {'ppo': {'observation_mode': 'position_only'}})
    no_b = PPO(no_b_config['ppo'], scaler)
    path = tmp_path / 'no_b.pt'
    save_checkpoint(path, no_b, no_b_config, {'episode': 1})
    payload = read_checkpoint(path)
    assert payload['config']['ppo']['observation_mode'] == 'position_only'
    restored = PPO(payload['config']['ppo'], scaler)
    restore_checkpoint(payload, restored)
    assert restored.observation_mode == 'position_only'
    with pytest.raises(ValueError, match='differs from policy'):
        restore_checkpoint(payload, PPO(config['ppo'], scaler))
    with pytest.raises(ValueError, match='differs from policy'):
        save_checkpoint(tmp_path / 'wrong.pt', no_b, config, {'episode': 1})
    legacy = deepcopy(payload)
    legacy['config']['ppo'].pop('observation_mode')
    legacy['format'] = 1
    restore_checkpoint(legacy, PPO(config['ppo'], scaler))
    with pytest.raises(ValueError, match='differs from policy'):
        restore_checkpoint(legacy, no_b)


def test_same_trajectory_preserves_environment_gp_and_rewards(config, small_field, scaler):
    config = merge(config, {'env': {'grid_size': 3, 'max_steps': 8}})
    agents = [PPO({**config['ppo'], 'observation_mode': mode}, scaler)
              for mode in ('full', 'position_only')]
    envs = [GridEnv(small_field, scaler, env_config(config)) for _ in agents]
    results = [env.reset(seed=1000, options={'start': [0, 0]}) for env in envs]
    for action in (3, 0, 2, 1, 4, 6, 5, 7):
        for agent, result in zip(agents, results):
            agent.prepare_state(result[0])
        results = [env.step(action) for env in envs]
        np.testing.assert_array_equal(results[0][0], results[1][0])
        assert results[0][1:4] == results[1][1:4]
        for key in results[0][4]:
            np.testing.assert_equal(results[0][4][key], results[1][4][key])
        np.testing.assert_array_equal(envs[0].gp.X, envs[1].gp.X)
        a = envs[0].gp.predict(envs[0].evaluation_points)
        b = envs[1].gp.predict(envs[1].evaluation_points)
        for first, second in zip(a, b):
            np.testing.assert_array_equal(first, second)


def test_scene_switch_preserves_checkpoint_mode(config, small_field, scaler, tmp_path):
    config = merge(config, {'ppo': {'observation_mode': 'position_only'},
                           'env': {'grid_size': 3, 'max_steps': 2},
                           'evaluation': {'seeds': [1000]}})
    path = tmp_path / 'policy.pt'
    save_checkpoint(path, PPO(config['ppo'], scaler), config, {'episode': 1})
    payload = read_checkpoint(path)
    for scenario in ('zhurong', 'perseverance'):
        switched = merge(payload['config'], {'field': {'test_scenario': scenario}})
        agent = PPO(switched['ppo'], scaler)
        restore_checkpoint(payload, agent)
        captured = []
        handle = agent.network.actor.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().cpu().numpy().copy()))
        evaluate(small_field, scaler, switched, agent, episode=1)
        handle.remove()
        assert len(captured) == 2 and all(np.all(state[2:] == 0) for state in captured)


def test_scaler_roundoff_compatibility_and_real_mismatch(scaler):
    from mars_ipp.training.checkpoint import scalers_equivalent
    stored = scaler.to_dict()
    perturbed = deepcopy(stored)
    perturbed['mean'][0] += 2e-12
    assert scalers_equivalent(stored, perturbed)
    perturbed['mean'][0] += .001
    assert not scalers_equivalent(stored, perturbed)
    perturbed = deepcopy(stored)
    perturbed['coordinate_scale'] += 1
    assert not scalers_equivalent(stored, perturbed)
    perturbed = deepcopy(stored)
    perturbed['std'][0] = float('nan')
    assert not scalers_equivalent(stored, perturbed)


def test_evaluation_uses_exact_checkpoint_scaler(config, small_field, scaler, tmp_path, monkeypatch):
    import importlib
    import json
    from mars_ipp.envs.observations import ObservationScaler
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    script = importlib.import_module('evaluate')
    config = merge(config, {'env': {'grid_size': 3, 'max_steps': 2},
                           'evaluation': {'seeds': [1000]}, 'ppo': {'observation_mode': 'position_only'}})
    stored = scaler.to_dict()
    stored['mean'][0] += 2e-12
    saved_scaler = ObservationScaler(**stored)
    checkpoint = tmp_path / 'checkpoint.pt'
    save_checkpoint(checkpoint, PPO(config['ppo'], saved_scaler), config, {'episode': 2})
    monkeypatch.setattr(script, 'prepare_fields', lambda config: {'zhurong': small_field, 'perseverance': small_field})
    original = script.evaluate

    def checked_evaluate(field, evaluation_scaler, config, agent, *args):
        assert evaluation_scaler is agent.scaler
        assert evaluation_scaler.to_dict() == stored
        return original(field, evaluation_scaler, config, agent, *args)

    monkeypatch.setattr(script, 'evaluate', checked_evaluate)
    monkeypatch.setattr(script, 'plot_results', lambda *args, **kwargs: None)
    output = tmp_path / 'evaluation'
    script.run(config, output, str(checkpoint))
    assert json.loads((output / 'metadata.json').read_text())['scaler'] == stored
    stored['mean'][0] += .001
    save_checkpoint(checkpoint, PPO(config['ppo'], ObservationScaler(**stored)), config, {'episode': 2})
    with pytest.raises(ValueError, match='normalization differs'):
        script.run(config, tmp_path / 'invalid', str(checkpoint))
