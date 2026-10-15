import numpy as np

ACTIONS = np.asarray([(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)], dtype=int)


def legal_neighbors(position, size, cardinal=False):
    candidates = np.asarray(position) + ACTIONS[:4 if cardinal else 8]
    mask = ((candidates >= 0) & (candidates < size)).all(axis=1)
    return np.flatnonzero(mask), candidates[mask]
