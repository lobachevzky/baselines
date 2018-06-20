import itertools

import numpy as np
from gym.envs.toy_text import FrozenLakeEnv
from gym.envs.toy_text.frozen_lake import MAPS


def s_connects_to_g(m, pos=(0, 0), shape=None):
    if shape is None:
        shape = np.shape(m)
    w, h = shape

    def in_bounds(x, y):
        return 0 <= x < w and 0 <= y < h

    def _connects(delta):
        new_pos = tuple(x + d for x, d in zip(pos, delta))
        if in_bounds(*new_pos) and not m[new_pos] in 'EH':
            if m[new_pos] == 'G':
                return True
            m[pos] = 'E'
            return s_connects_to_g(m, new_pos, shape=shape)
        return False

    return any(_connects(d) for d in
               itertools.product([0, 1], [0, 1]))


class FrozenLakeGoalEnv(FrozenLakeEnv):
    def __init__(self, desc=None, map_dims=(4, 4), is_slippery=True):
        h, w = map_dims
        map_name = f'{h}x{w}'
        self.map = np.random.choice(['F', 'H'], [h, w], p=[.85, .15])
        while True:
            m = np.random.choice(['F', 'H'], [h, w], p=[.85, .15])
            m[-1, -1] = 'G'
            if s_connects_to_g(m):
                print('Final Map:')
                np.set_printoptions(threshold=np.nan)
                print(m)
                np.set_printoptions(threshold=None)
                break
        m[0, 0] = 'S'
        m = [''.join(row) for row in m]
        MAPS[map_name] = m
        super().__init__(desc, map_name, is_slippery)

    # TODO random choice of start and goal
