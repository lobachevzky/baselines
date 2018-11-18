import numpy as np
import tensorflow as tf

from baselines.common.tf_util import get_session
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from environments import hindsight_wrapper as hw
from environments import hsr
from sac.utils import concat_spaces, space_rank, vectorize


class HSREnv(hsr.HSREnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Sadly, ppo code really likes boxes, so had to concatenate things
        spaces = hw.Observation(*self.observation_space.spaces)
        self.observation_space = concat_spaces(spaces, axis=1)

    def step(self, action):
        s, r, t, i = super().step(action)
        return vectorize(s), r, t, i

    def reset(self):
        return vectorize(super().reset())


class UnsupervisedEnv(hw.HSREnv):
    def __init__(self, seed: int, **kwargs):
        super().__init__(**kwargs)
        spaces = hw.Observation(*self.observation_space.spaces)

        # subspace_sizes used for splitting concatenated tensor observations
        self.subspace_sizes = space_rank(self.observation_space)
        for n in self.subspace_sizes:
            assert isinstance(n, int)

        # space of observation excluding concatenated reward param
        self.raw_observation_space = concat_spaces(
            [spaces.observation, spaces.achieved_goal], axis=1)

        # for defining reward param tf.Variable
        self.param_shape = spaces.desired_goal.low.shape

        # Sadly, ppo code really likes boxes, so had to concatenate things
        self.observation_space = concat_spaces(spaces, axis=1)

        self.rank = seed
        self.reward_params = None
        self.sess = get_session()

    def step(self, action):
        s, r, t, i = super().step(action)
        return vectorize([s.observation, s.achieved_goal]), r, t, i

    def reset(self):
        return vectorize(super().reset())

    def compute_reward(self):
        return -np.sum(np.square(self.reward_params - self.achieved_goal()))

    @staticmethod
    def reward_function(X: tf.Tensor, size_subspaces):
        with tf.variable_scope('reward', reuse=tf.AUTO_REUSE):
            param = tf.get_variable('params')
        achieved = UnsupervisedEnv.observation(
            *tf.split(X, size_subspaces, axis=1)).achieved_goal
        return -tf.reduce_sum(tf.square(achieved - param))

    def compute_terminal(self):
        return False

    def achieved_goal(self):
        return self.gripper_pos()

    def new_goal(self):
        return self.reward_params

    def set_reward_params(self, reward_params):
        self.reward_params = reward_params


class UnsupervisedVecEnv(SubprocVecEnv):
    def reset(self):
        self._assert_not_closed()
        with tf.variable_scope('reward', reuse=tf.AUTO_REUSE):
            params = tf.get_variable('params')
        for i, remote in enumerate(self.remotes):
            remote.send(('set_reward_params', params[i]))
        return super().reset()


class UnsupervisedDummyVecEnv(DummyVecEnv):
    def reset(self):
        with tf.variable_scope('reward', reuse=tf.AUTO_REUSE):
            params = tf.get_variable('params')
        for i, env in enumerate(self.envs):
            env.set_reward_params(params[i])
        return super().reset()
