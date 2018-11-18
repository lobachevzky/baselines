# third party
from environments import hsr
import numpy as np
from sac.utils import concat_spaces, space_shape, unwrap_env, vectorize
import tensorflow as tf

# first party
from baselines.common.tf_util import get_session
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


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


class UnsupervisedEnv(hsr.HSREnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        spaces = hsr.Observation(*self.observation_space.spaces)

        # subspace_sizes used for splitting concatenated tensor observations
        self.subspace_sizes = [space_shape(space)[0] for space in spaces]
        for n in self.subspace_sizes:
            assert isinstance(n, int)

        # space of observation needs to exclude reward param
        self.observation_space = concat_spaces(spaces, axis=0)

        self.reward_params = None
        self.sess = get_session()

    def step(self, action):
        s, r, t, i = super().step(action)
        return vectorize([s.observation, self.achieved_goal()]), r, t, i

    def reset(self):
        return vectorize(super().reset())

    def compute_reward(self):
        return -np.sum(np.square(self.reward_params - self.achieved_goal()))

    def compute_terminal(self):
        return False

    def achieved_goal(self):
        return self.gripper_pos()

    def new_goal(self):
        print('new goal params', self.reward_params)
        return self.reward_params

    def set_reward_params(self, reward_params):
        self.reward_params = reward_params


class UnsupervisedVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, reward_params: tf.Tensor):
        super().__init__(env_fns)
        self.params = reward_params

    def reset(self):
        self._assert_not_closed()
        params = get_session().run(self.params)
        for remote in self.remotes:
            remote.send(('set_reward_params', params))
        return super().reset()


class UnsupervisedDummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns, reward_params: tf.Tensor):
        super().__init__(env_fns)
        self.params = reward_params
        self.unwrapped_envs = [
            unwrap_env(env, lambda e: isinstance(e, UnsupervisedEnv))
            for env in self.envs
        ]

    def reset(self):
        params = get_session().run(self.params)
        print('run session params', params)
        for env in self.unwrapped_envs:
            env.set_reward_params(params)
        return super().reset()

    @property
    def param_shape(self):
        return self.unwrapped_envs[0].param_shape

    @property
    def raw_observation_space(self):
        return self.unwrapped_envs[0].raw_observation_space
