"""Online shared-kernel GP; only robot observations enter this posterior."""
import numpy as np
import GPy
from .kernels import make_kernel


class OnlineGP:
    def __init__(self, parameters, noise_variance=None, unique_samples=False):
        self.parameters = dict(parameters)
        self.kernel = make_kernel(parameters)
        self.noise = parameters['noise'] if noise_variance is None else float(noise_variance)
        if not np.isfinite(self.noise) or self.noise <= 0:
            raise ValueError('Online observation noise must be finite and positive')
        self.unique_samples = unique_samples
        self.jitter = 0.
        self.revision = 0
        self.X = np.empty((0, 2))
        self.Y = np.empty((0, 3))
        self.model = None

    def add(self, position, observation):
        x = np.asarray(position, dtype=float).reshape(1, 2)
        y = np.asarray(observation, dtype=float).reshape(1, 3)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite observation')
        if self.unique_samples and np.any(np.all(self.X == x, axis=1)):
            return False
        self.X = np.vstack((self.X, x))
        self.Y = np.vstack((self.Y, y))
        if self.model is None:
            self.model = GPy.models.GPRegression(self.X.copy(), self.Y.copy(), self.kernel, noise_var=self.noise)
            self.model.kern.fix()
            self.model.Gaussian_noise.variance.fix()
        else:
            self.model.set_XY(self.X.copy(), self.Y.copy())
        # Recover numerical diagonal additions without confusing them with likelihood noise.
        matrix = self.kernel.K(self.X) + np.eye(len(self.X)) * self.noise
        factor = self.model.posterior.woodbury_chol
        residual = factor @ factor.T - matrix
        self.jitter = max(0., float(np.diag(residual).mean()))
        if not np.isfinite(factor).all() or np.max(np.abs(residual)) > 1e-8 * max(1., float(self.kernel.variance[0])):
            raise FloatingPointError(f'Unexpected GP numerical jitter: {self.jitter}')
        self.revision += 1
        return True

    def predict(self, positions):
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        if self.model is None:
            return np.zeros((len(positions), 3)), self.kernel.Kdiag(positions)[:, None]
        mean, variance = self.model.predict_noiseless(positions)
        if variance.shape != (len(positions), 1):
            raise ValueError(f'Expected shared variance (N,1), got {variance.shape}')
        prior = self.kernel.Kdiag(positions)[:, None]
        tolerance = 1e-8 * np.maximum(prior, 1.)
        if (not np.isfinite(mean).all() or not np.isfinite(variance).all()
                or np.any(variance < -tolerance) or np.any(variance > prior + tolerance)):
            raise FloatingPointError('Invalid GP latent posterior variance')
        return mean, np.clip(variance, 0., prior)

    def covariance(self, left, right):
        left, right = np.asarray(left), np.asarray(right)
        cov = self.kernel.K(left, right)
        if self.model is not None:
            cov -= self.kernel.K(left, self.X) @ self.model.posterior.woodbury_inv @ self.kernel.K(self.X, right)
        return cov
