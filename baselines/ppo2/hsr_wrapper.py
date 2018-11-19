# third party
from collections import namedtuple

import numpy as np
import tensorflow as tf

# first party
from baselines.common.tf_util import get_session
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from environments import hsr
from sac.utils import concat_spaces, space_shape, vectorize, unwrap_env


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

        # self.reward_params = self.achieved_goal()
        self.reward_params = 7 * np.ones(3)
        self.sess = get_session()

    def step(self, actions):
        s, r, t, i = super().step(actions)
        print('step')
        print('self.goal', self.goal)
        print('s.goal', s.goal)
        return vectorize(
            Observation(
                observation=s.observation,
                params=s.goal,
                achieved=self.achieved_goal())), r, t, i

    def reset(self):
        o = super().reset()
        n = vectorize(Observation(observation=o.observation, params=o.goal, achieved=self.achieved_goal()))
        return n

    def compute_reward(self):
        return -np.sum(np.square(self.reward_params - self.achieved_goal()))

    def compute_terminal(self):
        return False

    def achieved_goal(self):
        return self.gripper_pos()

    def new_goal(self):
        print('new goal params', self.reward_params)
        return self.reward_params

    def set_reward_params(self, param):
        self.reward_params = param
        self.set_goal(param)


class UnsupervisedSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, reward_params: tf.Tensor):
        super().__init__(env_fns)
        self.params = reward_params
        self.sess = get_session()

    def set_reward_params(self):
        params = self.sess.run(self.params)
        for remote, param in zip(self.remotes, params):
            remote.send(('set_reward_params', param))

class UnsupervisedDummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns, reward_params: tf.Tensor):
        super().__init__(env_fns)
        self.params = reward_params
        self.sess = get_session()

    def set_reward_params(self):
        params = self.sess.run(self.params)
        for env, param in zip(self.envs, params):
            _env = unwrap_env(env, lambda e: hasattr(e, 'set_reward_params'))
            _env.set_reward_params(param)


