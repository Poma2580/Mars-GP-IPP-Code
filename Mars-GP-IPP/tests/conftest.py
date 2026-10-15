import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import pytest
from threadpoolctl import threadpool_limits
import GPy
from mars_ipp.field.field_model import FieldModel
from mars_ipp.envs.observations import ObservationScaler
from mars_ipp.utils.config import ROOT, load_config


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def config():
    return load_config(ROOT / 'configs/smoke.yaml')


@pytest.fixture
def small_field():
    import numpy as np
    x = np.array([[0., 0.], [0., 4.], [4., 0.], [4., 4.]])
    y = np.column_stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1], 2*x[:, 1]))
    return FieldModel(GPy.models.GPRegression(x, y, GPy.kern.RBF(2, variance=2., lengthscale=2.), noise_var=.1))


@pytest.fixture
def scaler(small_field):
    return ObservationScaler.from_training_field(small_field, 3, 2.)
