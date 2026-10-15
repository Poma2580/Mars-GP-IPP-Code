import numpy as np
from mars_ipp.envs.actions import ACTIONS


class Coverage:
    start = (14, 14)

    def __init__(self, seed=0, steps=70, random_start=False, size=16):
        self.random_start, self.size = random_start, size
        self.horizontal = self.vertical = -1
        self.turn_remaining = 0
        self.turn_horizontal = 0
        self.actions = []
        left = True
        while len(self.actions) < steps:
            self.actions.extend(([3] * 13 + [6, 7]) if left else ([1] * 13 + [7, 6]))
            left = not left
        self.actions = self.actions[:steps]
        self.index = 0

    def path(self):
        return np.vstack((self.start, np.asarray(self.start) + np.cumsum(ACTIONS[self.actions], axis=0)))

    def select_action(self, observation, planning_context=None):
        if self.random_start:
            x, y = np.asarray(observation[:2], dtype=int)
            if not self.turn_remaining and ((self.horizontal < 0 and x <= 1) or (self.horizontal > 0 and x >= self.size - 2)):
                self.turn_remaining = 2
                self.turn_horizontal = self.horizontal if 0 <= x + self.horizontal < self.size else -self.horizontal
            if self.turn_remaining:
                dx = self.turn_horizontal if self.turn_remaining == 2 else -self.turn_horizontal
                if not 0 <= y + self.vertical < self.size:
                    self.vertical *= -1
                move = (dx, self.vertical)
                self.turn_remaining -= 1
                if not self.turn_remaining:
                    self.horizontal *= -1
            else:
                move = (self.horizontal, 0)
            return int(np.flatnonzero(np.all(ACTIONS == move, axis=1))[0])
        action = self.actions[self.index]
        self.index += 1
        return action
