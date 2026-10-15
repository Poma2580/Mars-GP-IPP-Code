"""Single-point entropy is monotonic in variance; same policy as GP-Greedy."""
import numpy as np
from .gp_greedy import GPGreedy


class GPEntropy(GPGreedy):
    def entropy(self, context):
        return 0.5 * np.log(2 * np.pi * np.e * np.maximum(self.scores(context), np.finfo(float).tiny))

    # Inherit selection in variance space to keep tie treatment exactly equivalent.
