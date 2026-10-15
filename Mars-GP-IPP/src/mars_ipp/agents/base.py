from dataclasses import dataclass
from typing import Protocol
import numpy as np
from mars_ipp.gp.estimator import OnlineGP


@dataclass(frozen=True)
class PlanningContext:
    position: np.ndarray
    actions: np.ndarray
    candidate_positions: np.ndarray
    gp: OnlineGP
    evaluation_points: np.ndarray


class Agent(Protocol):
    def select_action(self, observation, planning_context: PlanningContext) -> int: ...
