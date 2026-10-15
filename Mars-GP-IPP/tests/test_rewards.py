import numpy as np
import pytest
from mars_ipp.envs.rewards import reward_terms, weighted_reward
from mars_ipp.envs.observations import ObservationScaler
from mars_ipp.envs.actions import legal_neighbors


def test_rewards_and_ablation():
    terms = reward_terms([0, 0, 0], [1, 2, 3], 3, 10, 5, repeated=True)
    assert terms == pytest.approx({'mh': 1., 'tr': .5, 'rep': -1., 'out': 0.})
    assert weighted_reward(terms, [.25]*4) == pytest.approx(.125)
    disabled = reward_terms([0, 0, 0], [1, 2, 3], 3, 10, 5, repeated=True, disabled=['mh'])
    assert weighted_reward(disabled, [.25]*4) == pytest.approx(-.125)
    zero = reward_terms([0]*3, [0]*3, 0, 0, 0)
    assert zero['mh'] == zero['tr'] == 0
    outside = reward_terms([0]*3, [9]*3, 1, 2, 0, outside=True)
    assert outside == {'mh': 0., 'tr': 0., 'rep': 0., 'out': -1.}
    assert reward_terms([0]*3, [1]*3, 2, 1, 2)['tr'] == 0


def test_d_ref_and_training_statistics(small_field, scaler):
    points = np.array([(i,j) for i in range(3) for j in range(3)])
    values = small_field.truth(points * 2)
    np.testing.assert_allclose(scaler.mean, values.mean(axis=0))
    np.testing.assert_allclose(scaler.std, values.std(axis=0))
    distances = [np.abs(small_field.truth([q*2])[0]-small_field.truth([p*2])[0]).sum()
                 for p in points for q in legal_neighbors(p, 3)[1]]
    assert scaler.d_ref == pytest.approx(max(distances))
