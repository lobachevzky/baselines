# third party
from collections import namedtuple

import numpy as np
import tensorflow as tf

# first party
from baselines.common.tf_util import get_session
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from environments import hsr
from sac.utils import concat_spaces, space_shape, vectorize


class HSREnv(hsr.HSREnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Sadly, ppo code really likes boxes, so had to concatenate things
        self.observation_space = concat_spaces(
            self.observation_space.spaces, axis=0)

    def step(self, action):
        s, r, t, i = super().step(action)
        return vectorize(s), r, t, i

    def reset(self):
        return vectorize(super().reset())


class MoveGripperEnv(hsr.HSREnv, hsr.MoveGripperEnv):
    pass

StepData = namedtuple('StepData', 'actions reward_params')


class Observation(namedtuple('Observation', 'observation achieved params')):
    def replace(self, *args, **kwargs):
        return self._replace(*args, **kwargs)


class UnsupervisedEnv(hsr.HSREnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        old_spaces = hsr.Observation(*self.observation_space.spaces)
        spaces = Observation(
            observation=old_spaces.observation,
            params=old_spaces.goal,
            achieved=old_spaces.goal)

        # subspace_sizes used for splitting concatenated tensor observations
        self.subspace_sizes = [space_shape(space)[0] for space in spaces]
        for n in self.subspace_sizes:
            assert isinstance(n, int)

        # space of observation needs to exclude reward param
        self.observation_space = concat_spaces(spaces, axis=0)

        self.reward_params = spaces.params.sample()
        self.sess = get_session()

    def step(self, step_data: StepData):
        self.reward_params = step_data.reward_params
        s, r, t, i = super().step(step_data.actions)
        return vectorize([s, self.achieved_goal()]), r, t, i

    def reset(self):
        o = super().reset()
        return vectorize([o, self.achieved_goal()])

    def compute_reward(self):
        return -np.sum(np.square(self.reward_params - self.achieved_goal()))

    def compute_terminal(self):
        return False

    def achieved_goal(self):
        return self.gripper_pos()

    def new_goal(self):
        print('new goal params', self.reward_params)
        return self.reward_params


class UnsupervisedSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, reward_params: tf.Tensor):
        super().__init__(env_fns)
        self.params = reward_params
        self.sess = get_session()

    def step_async(self, actions):
        params = self.sess.run(self.params)
        super().step_async([
            StepData(actions=action, reward_params=params[i])
            for i, action in enumerate(actions)
        ])


class UnsupervisedDummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns, reward_params: tf.Tensor):
        super().__init__(env_fns)
        self.params = reward_params
        self.sess = get_session()

    def step_async(self, actions):
        params = self.sess.run(self.params)
        super().step_async([
            StepData(actions=action, reward_params=params[i])
            for i, action in enumerate(actions)
        ])
