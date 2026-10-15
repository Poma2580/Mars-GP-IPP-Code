import numpy as np


class GPGreedy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def scores(self, context):
        return context.gp.predict(context.candidate_positions)[1][:, 0]

    def select_action(self, observation, planning_context):
        scores = self.scores(planning_context)
        ties = np.flatnonzero(np.isclose(scores, scores.max(), rtol=1e-10, atol=1e-12))
        return int(planning_context.actions[self.rng.choice(ties)])
