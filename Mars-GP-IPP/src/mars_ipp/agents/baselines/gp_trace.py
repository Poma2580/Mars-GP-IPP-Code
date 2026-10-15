from .gp_greedy import GPGreedy
from mars_ipp.gp.uncertainty import expected_trace_reduction


class GPTrace(GPGreedy):
    def scores(self, context):
        return expected_trace_reduction(context.gp, context.candidate_positions, context.candidate_positions)
