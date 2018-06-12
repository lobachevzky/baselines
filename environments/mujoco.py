import os
from abc import abstractmethod
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

import mujoco


class MujocoEnv:
    def __init__(self, xml_filepath: Path,
                 image_dimensions: Optional[Tuple[int]],
                 neg_reward: bool,
                 steps_per_action: int,
                 render_freq: int):
        if not xml_filepath.is_absolute():
            xml_filepath = Path(Path(__file__).parent, xml_filepath)
        self.sim = mujoco.Sim(str(xml_filepath), n_substeps=1)
        self.init_qpos = self.sim.qpos.ravel().copy()
        self.init_qvel = self.sim.qvel.ravel().copy()
        self._step_num = 0
        self._neg_reward = neg_reward
        self._image_dimensions = image_dimensions

        self.observation_space = self.action_space = None

        # required for OpenAI code
        self.metadata = {'render.modes': 'rgb_array'}
        self.reward_range = -np.inf, np.inf
        self.spec = None
        self.render_freq = render_freq
        self.steps_per_action = steps_per_action

    def seed(self, seed=None):
        np.random.seed(seed)

    def server_values(self):
        return self.sim.qpos, self.sim.qvel

    def render(self, mode=None, camera_name=None, labels=None):
        if mode == 'rgb_array':
            return self.sim.render_offscreen(height=256, width=256)
        if labels is None:
            labels = dict(x=self.goal_3d())
        self.sim.render(camera_name, labels)

    def image(self, camera_name='rgb'):
        return self.sim.render_offscreen(*self._image_dimensions, camera_name)

    def step(self, action):
        assert np.shape(action) == np.shape(self.sim.ctrl)
        self._set_action(action)
        done = self.compute_terminal(self.goal(), self._get_obs())
        reward = self.compute_reward(self.goal(), self._get_obs())
        return self._get_obs(), reward, done, {}

    def _set_action(self, action):
        assert np.shape(action) == np.shape(self.sim.ctrl)
        for i in range(self.steps_per_action):
            self.sim.ctrl[:] = action
            self.sim.step()
            if self.render_freq > 0 and i % self.render_freq == 0:
                self.render()

    def reset(self):
        self.sim.reset()
        self._step_num = 0

        self._set_new_goal()
        qpos = self.reset_qpos()
        qvel = self.init_qvel + \
            np.random.uniform(size=self.sim.nv, low=-0.01, high=0.01)
        assert qpos.shape == (self.sim.nq, ) and qvel.shape == (self.sim.nv, )
        self.sim.qpos[:] = qpos.copy()
        self.sim.qvel[:] = qvel.copy()
        self.sim.forward()
        return self._get_obs()

    @abstractmethod
    def reset_qpos(self):
        raise NotImplemented

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.sim.__exit__()

    @abstractmethod
    def _set_new_goal(self):
        raise NotImplementedError

    @abstractmethod
    def _get_obs(self):
        raise NotImplementedError

    @abstractmethod
    def goal(self):
        raise NotImplementedError

    @abstractmethod
    def goal_3d(self):
        raise NotImplementedError

    @abstractmethod
    def compute_terminal(self, goal, obs):
        raise NotImplementedError

    @abstractmethod
    def compute_reward(self, goal, obs):
        raise NotImplementedError


def quaternion2euler(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    euler_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = 1 if t2 > 1 else t2
    t2 = -1 if t2 < -1 else t2
    euler_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    euler_z = np.arctan2(t3, t4)

    return euler_x, euler_y, euler_z


def distance_between(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2), axis=-1))


def at_goal(pos, goal, geofence, verbose=False):
    distance_to_goal = distance_between(pos, goal)
    if verbose:
        print(distance_to_goal)
    return distance_to_goal < geofence


def escaped(pos, world_upper_bound, world_lower_bound):
    # noinspection PyTypeChecker
    return np.any(pos > world_upper_bound) \
        or np.any(pos < world_lower_bound)


def get_limits(pos, size):
    return pos + size, pos - size


def point_inside_object(point, object):
    pos, size = object
    tl = pos - size
    br = pos + size
    return (tl[0] <= point[0] <= br[0]) and (tl[1] <= point[1] <= br[1])


def print1(*strings):
    print('\r', *strings, end='')
