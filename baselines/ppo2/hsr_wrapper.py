from gym.spaces import Box
import numpy as np

from environments import hsr
from sac.utils import space_to_size, vectorize


class HSREnv(hsr.HSREnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=[space_to_size(self.observation_space)])

    def step(self, action):
        s, r, t, i = super().step(action)
        return vectorize(s), r, t, i

    def reset(self):
        return vectorize(super().reset())


class MoveGripperEnv(hsr.MoveGripperEnv, HSREnv):
    pass
