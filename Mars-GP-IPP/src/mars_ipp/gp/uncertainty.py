import numpy as np


def trace(variance):
    variance = np.asarray(variance)
    if variance.ndim != 2 or variance.shape[1] != 1:
        raise ValueError('Trace requires shared variance of shape (N,1), not three copies')
    return float(variance.sum())


def expected_trace_reduction(gp, candidates, evaluation_points):
    cross = gp.covariance(evaluation_points, candidates)
    _, variance = gp.predict(candidates)
    denominator = variance[:, 0] + gp.noise
    return np.sum(cross ** 2, axis=0) / np.maximum(denominator, np.finfo(float).eps)
