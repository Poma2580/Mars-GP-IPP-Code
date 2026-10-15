import numpy as np
import pytest
from mars_ipp.gp.estimator import OnlineGP
from mars_ipp.gp.uncertainty import trace
from mars_ipp.field.dataset import evaluation_grid


def test_prior_shared_variance_and_robot_only_posterior(small_field):
    gp = OnlineGP(small_field.parameters)
    grid = evaluation_grid()
    mean, var = gp.predict(grid)
    assert mean.shape == (961, 3) and var.shape == (961, 1)
    assert len(gp.X) == 0
    assert np.all(mean == 0)
    assert trace(var) == pytest.approx(961 * 2)
    gp.add([0, 0], [1, 2, 3])
    assert gp.X.shape == (1, 2) and gp.Y.shape == (1, 3)
    assert trace(gp.predict(grid)[1]) < trace(var)
    assert gp.model.kern.is_fixed and gp.model.Gaussian_noise.variance.is_fixed
    parameters = gp.model.param_array.copy()
    gp.add([0, 0], [1, 2, 3])
    np.testing.assert_array_equal(parameters, gp.model.param_array)
    np.testing.assert_allclose(gp.predict(grid)[1], gp.model.predict(grid)[1] - gp.noise, atol=1e-10)


def test_grid_order_and_extents():
    grid = evaluation_grid()
    np.testing.assert_equal(grid[0], [0, 0])
    np.testing.assert_equal(grid[30], [30, 0])
    np.testing.assert_equal(grid[31], [0, 1])
    np.testing.assert_equal(grid[-1], [30, 30])
