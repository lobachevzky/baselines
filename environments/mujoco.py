from abc import abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import mujoco


class MujocoEnv:
    def __init__(self,
                 xml_filepath: Path,
                 steps_per_action: int,
                 image_dimensions: Optional[Tuple[int]] = None,
                 record_path: Optional[Path] = None,
                 record_freq: int = 0,
                 record: bool = None,
                 render_freq: int = 0):
        if not xml_filepath.is_absolute():
            xml_filepath = Path(Path(__file__).parent, xml_filepath)

        self.observation_space = self.action_space = None

        # required for OpenAI code
        self.metadata = {'render.modes': 'rgb_array'}
        self.reward_range = -np.inf, np.inf
        self.spec = None
        self.render_freq = render_freq
        self.steps_per_action = steps_per_action

        self.video_recorder = None
        self._record_video = any((record_path, record_freq, record))
        if self._record_video:
            if not record_path:
                record_path = Path('/tmp/training-video')
            if not image_dimensions:
                image_dimensions = (400, 400)
            if not record_freq:
                record_freq = 20

            print(f'Recording video to {record_path}.mp4')
            record_path.mkdir(exist_ok=True)
            self._record_freq = record_freq
            self._image_dimensions = image_dimensions

            self.video_recorder = VideoRecorder(
                env=self,
                base_path=str(record_path),
                enabled=True,
            )
        else:
            image_dimensions = image_dimensions or []

        self.sim = mujoco.Sim(str(xml_filepath), *image_dimensions, n_substeps=1)

        self.init_qpos = self.sim.qpos.ravel().copy()
        self.init_qvel = self.sim.qvel.ravel().copy()
        self._image_dimensions = image_dimensions

    def seed(self, seed=None):
        np.random.seed(seed)

    def server_values(self):
        return self.sim.qpos, self.sim.qvel

    def render(self, mode=None, camera_name=None, labels=None):
        if mode == 'rgb_array':
            return self.sim.render_offscreen(camera_name=camera_name)
        return self.sim.render(camera_name=camera_name, labels=labels)

    def image(self, camera_name='rgb'):
        return self.sim.render_offscreen(camera_name)

    def step(self, action):
        assert np.shape(action) == np.shape(self.sim.ctrl)
        self._set_action(action)
        done = self.compute_terminal()
        reward = self.compute_reward()
        return self._get_obs(), reward, done, {}

    def _set_action(self, action):
        assert np.shape(action) == np.shape(self.sim.ctrl)
        for i in range(self.steps_per_action):
            self.sim.ctrl[:] = action
            self.sim.step()
            if self.render_freq > 0 and i % self.render_freq == 0:
                self.render()
            if self._record_video and i % self._record_freq == 0:
                self.video_recorder.capture_frame()

    def reset(self):
        self.sim.reset()
        qpos = self._reset_qpos()
        assert qpos.shape == (self.sim.nq, )
        self.sim.qpos[:] = qpos.copy()
        self.sim.qvel[:] = 0
        self.sim.forward()
        return self._get_obs()

    @abstractmethod
    def _reset_qpos(self):
        raise NotImplemented

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.sim.__exit__()

    @abstractmethod
    def _get_obs(self):
        raise NotImplementedError

    @abstractmethod
    def compute_terminal(self):
        raise NotImplementedError

    @abstractmethod
    def compute_reward(self):
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
