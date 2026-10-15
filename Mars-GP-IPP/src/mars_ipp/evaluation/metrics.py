import numpy as np
from mars_ipp.gp.uncertainty import trace


def metrics(mean, variance, reference):
    mean, reference = np.asarray(mean), np.asarray(reference)
    if mean.shape != reference.shape or mean.ndim != 2 or mean.shape[1] != 3:
        raise ValueError('Mean and reference must both have shape (N,3)')
    error = (mean - reference) ** 2
    return {'rmse': float(np.sqrt(error.mean())), 'trace_current': trace(variance),
            **{f'rmse_{name}': float(np.sqrt(error[:, i].mean())) for i, name in enumerate(('Br', 'B2', 'B3'))}}
