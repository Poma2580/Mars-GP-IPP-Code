import numpy as np
import pytest
from mars_ipp.evaluation.metrics import metrics
from mars_ipp.gp.uncertainty import trace


def test_metrics_physical_scale_and_shared_trace():
    mean = np.tile([1., 2., 3.], (961, 1))
    result = metrics(mean, np.full((961,1), 2.), np.zeros_like(mean))
    assert result['rmse'] == pytest.approx(np.sqrt(14/3))
    assert result['rmse_Br'] == 1 and result['rmse_B2'] == 2 and result['rmse_B3'] == 3
    assert result['trace_current'] == 1922
    with pytest.raises(ValueError):
        trace(np.ones((961, 3)))
