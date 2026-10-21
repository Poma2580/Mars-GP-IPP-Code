"""Mask-only configuration isolation and restored policy compatibility."""
from copy import deepcopy
from unittest.mock import patch
import numpy as np
import pytest
from mars_ipp.experiments.v3_3_3_masked import configs, METHODS
from mars_ipp.experiments.v3_variants import validate_config, scaler_for, VariantEnv
from mars_ipp.experiments.v3_pipeline import load_policy
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.training.checkpoint import save_checkpoint


def test_only_mask_differs():
    base, new = configs()
    expected = deepcopy(base)
    expected['experiment'] = 'V3_3_3_m'
    expected['ppo']['action_mask'] = True
    assert new == expected
    assert new['reward']['weights'] == [1., .005]
    assert METHODS == ('V3.3.3', 'V3.3.3_m')
    assert 'action_mask' not in base['ppo']
    bad = deepcopy(new); bad['ppo']['action_mask'] = False
    with pytest.raises(ValueError, match='action_mask'):
        validate_config(bad)
    bad = deepcopy(new); bad['reward']['weights'] = [1., .014]
    with pytest.raises(ValueError, match='reward'):
        validate_config(bad)
    bad = deepcopy(new); bad['ppo']['actor_lr'] *= 2
    with patch('mars_ipp.experiments.v3_3_3_masked.load_config', side_effect=[base, bad]):
        with pytest.raises(ValueError, match='Only experiment'):
            configs()


def test_masked_reward_and_restore(small_field, tmp_path):
    _, c = configs(); c['ppo']['device'] = 'cpu'
    env = VariantEnv(small_field, scaler_for(c), c)
    obs, _ = env.reset(options={'start': [2, 2]})
    assert obs.shape == (11,)
    for action in [1, 4, 6]:
        _, reward, _, _, info = env.step(action)
        assert reward == pytest.approx(info['r_gp'] + .005*info['r_rep'])
        assert reward == pytest.approx(info['weighted_r_gp'] + info['weighted_r_rep'])
    assert info['r_rep'] == -1 and reward == -.005
    assert 'r_mh' not in info
    agent = PPO(c['ppo'], scaler_for(c))
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, agent, c, dict(completed=True))
    restored, _ = load_policy(path)
    assert restored.action_mask_enabled and restored.input_dim == 11
    obs = np.r_[[0, 0], np.ones(9)]
    for _ in range(100):
        assert restored.mask_for(obs)[restored.act(obs)[0]]
