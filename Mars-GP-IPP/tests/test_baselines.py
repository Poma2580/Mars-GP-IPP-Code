import numpy as np
import pytest
from mars_ipp.agents.baselines.gp_greedy import GPGreedy
from mars_ipp.agents.baselines.gp_entropy import GPEntropy
from mars_ipp.agents.baselines.gp_trace import GPTrace
from mars_ipp.agents.baselines.msr import Msr
from mars_ipp.agents.baselines.coverage import Coverage
from mars_ipp.gp.estimator import OnlineGP
from mars_ipp.gp.uncertainty import expected_trace_reduction, trace
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.evaluation.evaluator import env_config


def test_trace_analytic_against_explicit_update(small_field):
    gp = OnlineGP(small_field.parameters)
    gp.add([0,0], [1,2,3])
    grid = np.array([[0.,0.], [1.,1.], [3.,3.]])
    candidates = np.array([[1.,0.], [2.,2.]])
    predicted = expected_trace_reduction(gp, candidates, grid)
    before = trace(gp.predict(grid)[1])
    for i, point in enumerate(candidates):
        explicit = OnlineGP(small_field.parameters)
        explicit.add([0,0], [1,2,3])
        explicit.add(point, [99, -20, 3])  # Values cannot affect posterior variance.
        actual = before - trace(explicit.predict(grid)[1])
        assert predicted[i] == pytest.approx(actual, abs=1e-7)


def test_entropy_equivalence_and_legal_actions(config, small_field, scaler):
    env = GridEnv(small_field, scaler, {**env_config(config), 'grid_size': 3})
    obs, _ = env.reset(seed=3, options={'start': [0,0]})
    context = env.planning_context()
    greedy, entropy = GPGreedy(2), GPEntropy(2)
    for _ in range(10):
        assert greedy.select_action(obs, context) == entropy.select_action(obs, context)
    assert GPTrace(3).select_action(obs, context) in context.actions
