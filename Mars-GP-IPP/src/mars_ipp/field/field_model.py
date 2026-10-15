"""Offline truth field. Never pass this object into a planning context."""
import hashlib
import pickle
import numpy as np
from mars_ipp.gp.kernels import model_parameters


class FieldModel:
    def __init__(self, model, source=None):
        self.model = model
        self.parameters = model_parameters(model)
        self.source = str(source) if source else None
        self.sha256 = hashlib.sha256(source.read_bytes()).hexdigest() if source else None

    @classmethod
    def load(cls, path):
        # Only load trusted project artifacts: pickle can execute code.
        with path.open('rb') as stream:
            return cls(pickle.load(stream), path)

    def truth(self, positions):
        mean, _ = self.model.predict_noiseless(np.asarray(positions, dtype=float).reshape(-1, 2))
        if mean.shape[1] != 3:
            raise ValueError('Expected three magnetic components')
        return mean

    def measure(self, position, rng, noise_rate=0.):
        value = self.truth(np.asarray(position).reshape(1, 2))[0]
        if noise_rate:
            value = value * (1 + rng.uniform(-noise_rate, noise_rate, 3))
        return value
