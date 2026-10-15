import numpy as np
import pytest
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.evaluation.evaluator import env_config


def test_reset_bounds_revisits_and_termination(config, small_field, scaler):
    settings = env_config(config)
    settings.update(grid_size=3, max_steps=4)
    env = GridEnv(small_field, scaler, settings)
    obs, info = env.reset(seed=2, options={'start': [0, 0]})
    assert obs.shape == (5,) and env.count == 0 and len(env.gp.X) == 1
    assert info['stage'] == 'reset' and env.prior_info['samples'] == 0
    np.testing.assert_allclose(obs[2:], small_field.truth([[0, 0]])[0])
    next_obs, reward, done, truncated, info = env.step(3)
    np.testing.assert_array_equal(obs, next_obs)
    assert info['r_out'] == -1 and info['r_rep'] == 0 and not info['sampled']
    assert info['samples'] == 1 and not done and not truncated
    env.step(1)
    _, _, _, _, info = env.step(3)
    assert info['samples'] == 3 and info['r_rep'] == -1 and info['sampled']
    _, _, done, truncated, _ = env.step(0)
    assert done and not truncated
    with pytest.raises(RuntimeError):
        env.step(0)


def test_local_reward_uses_same_cardinal_points(config, small_field, scaler):
    env = GridEnv(small_field, scaler, {**env_config(config), 'grid_size': 3})
    env.reset(seed=1, options={'start': [0, 0]})
    # Destination (1,0): legal cardinal positions (1,1), (2,0), (0,0).
    points = np.array([[2, 2], [4, 0], [0, 0]])
    before = env.gp.predict(points)[1].sum()
    _, _, _, _, info = env.step(1)
    after = env.gp.predict(points)[1].sum()
    assert info['r_tr'] == pytest.approx((before-after)/(before+1e-12))
    context = env.planning_context()
    assert not hasattr(context, 'field') and not hasattr(context, 'reference')


def test_scaling_fixed_and_noise_reproducible(config, small_field, scaler):
    raw = np.array([2, 2, *scaler.mean])
    np.testing.assert_allclose(scaler.transform(raw), [1, 1, 0, 0, 0], atol=1e-7)
    raw[2:] += 100
    assert scaler.transform(raw)[2:].max() > 1  # No test-time clipping/refitting.
    settings = {**env_config(config), 'grid_size': 3, 'noise_rate': .3}
    a, b = GridEnv(small_field, scaler, settings), GridEnv(small_field, scaler, settings)
    np.testing.assert_equal(a.reset(seed=55)[0], b.reset(seed=55)[0])
    np.testing.assert_equal(a.step(0)[0], b.step(0)[0])


@pytest.mark.parametrize('noise', [0., .3])
def test_metric_modes_equivalent_without_intermediate_global_predictions(config, small_field, scaler, monkeypatch, noise):
    from mars_ipp.gp.estimator import OnlineGP
    settings = {**env_config(config), 'grid_size': 3, 'max_steps': 70, 'noise_rate': noise}
    full = GridEnv(small_field, scaler, settings)
    terminal = GridEnv(small_field, scaler, settings, metrics_mode='terminal')
    calls = {id(full): 0, id(terminal): 0}
    original_predict = OnlineGP.predict

    def predict(gp, points):
        if len(points) == 961:
            for env in (full, terminal):
                if gp is getattr(env, 'gp', None):
                    calls[id(env)] += 1
        return original_predict(gp, points)

    monkeypatch.setattr(OnlineGP, 'predict', predict)
    a, ai = full.reset(seed=27, options={'start': [0, 0]})
    b, bi = terminal.reset(seed=27, options={'start': [0, 0]})
    np.testing.assert_array_equal(a, b)
    assert 'rmse' not in bi and 'rmse' not in terminal.prior_info
    assert calls[id(terminal)] == 0
    assert ai['trace_prior'] == bi['trace_prior']
    for action in ([3, 1, 3, 0, 4, 7, 2] * 10):
        a, ar, ad, at, ai = full.step(action)
        b, br, bd, bt, bi = terminal.step(action)
        np.testing.assert_array_equal(a, b)
        assert (ar, ad, at) == (br, bd, bt)
        for key in ('r_mh', 'r_tr', 'r_rep', 'r_out', 'samples', 'sampled'):
            assert ai[key] == bi[key]
        np.testing.assert_array_equal(full.gp.X, terminal.gp.X)
        np.testing.assert_array_equal(full.gp.Y, terminal.gp.Y)
        if not bd:
            assert 'rmse' not in bi and calls[id(terminal)] == 0
    assert calls[id(full)] == 72 and calls[id(terminal)] == 1
    for key in ('rmse', 'trace_current', 'rmse_Br', 'rmse_B2', 'rmse_B3'):
        assert ai[key] == pytest.approx(bi[key], rel=1e-12, abs=1e-12)
    for left, right in zip(full.gp.predict(full.evaluation_points), terminal.gp.predict(terminal.evaluation_points)):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)


def test_invalid_metric_mode(config, small_field, scaler):
    with pytest.raises(ValueError, match='metrics_mode'):
        GridEnv(small_field, scaler, env_config(config), metrics_mode='invalid')
