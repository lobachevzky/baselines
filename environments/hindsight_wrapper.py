from abc import abstractmethod
from collections import namedtuple
from copy import deepcopy
from typing import Optional

import gym
import numpy as np
from gym.spaces import Box

from environments.mujoco import distance_between
from environments.multi_task import MultiTaskEnv
from environments.pick_and_place import PickAndPlaceEnv
from sac.array_group import ArrayGroup
from sac.utils import Step, unwrap_env, vectorize

Goal = namedtuple('Goal', 'gripper block')


class Observation(namedtuple('Obs', 'observation achieved_goal desired_goal')):
    def replace(self, **kwargs):
        return super()._replace(**kwargs)


class HindsightWrapper(gym.Wrapper):
    @abstractmethod
    def _achieved_goal(self):
        raise NotImplementedError

    @abstractmethod
    def _is_success(self, achieved_goal, desired_goal):
        raise NotImplementedError

    @abstractmethod
    def _desired_goal(self):
        raise NotImplementedError

    def step(self, action):
        o2, r, t, info = self.env.step(action)
        new_o2 = Observation(
            observation=o2,
            desired_goal=self._desired_goal(),
            achieved_goal=self._achieved_goal())
        return new_o2, r, t, info

    def reset(self):
        return Observation(
            observation=self.env.reset(),
            desired_goal=self._desired_goal(),
            achieved_goal=self._achieved_goal())

    def recompute_trajectory(self, trajectory: Step):
        trajectory = Step(*deepcopy(trajectory))

        # get values
        o1 = Observation(*trajectory.o1)
        o2 = Observation(*trajectory.o2)
        achieved_goal = ArrayGroup(o2.achieved_goal)[-1]

        # perform assignment
        ArrayGroup(o1.desired_goal)[:] = achieved_goal
        ArrayGroup(o2.desired_goal)[:] = achieved_goal
        trajectory.r[:] = self._is_success(o2.achieved_goal, o2.desired_goal)
        trajectory.t[:] = np.logical_or(trajectory.t, trajectory.r)

        first_terminal = np.flatnonzero(trajectory.t)[0]
        return ArrayGroup(trajectory)[:first_terminal + 1]  # include first terminal

    def preprocess_obs(self, obs, shape: Optional[tuple] = None):
        obs = Observation(*obs)
        obs = [obs.observation, obs.desired_goal]
        return vectorize(obs, shape=shape)


class MountaincarHindsightWrapper(HindsightWrapper):
    """
    new obs is [pos, vel, goal_pos]
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=vectorize([self.observation_space.low, env.unwrapped.min_position]),
            high=vectorize([self.observation_space.high, env.unwrapped.max_position]))

    def step(self, action):
        o2, r, t, info = super().step(action)
        is_success = self._is_success(o2.achieved_goal, o2.desired_goal)
        new_t = is_success or t
        new_r = float(is_success)
        info['base_reward'] = r
        return o2, new_r, new_t, info

    def _achieved_goal(self):
        return self.env.unwrapped.state[0]

    def _desired_goal(self):
        return 0.45

    def _is_success(self, achieved_goal, desired_goal):
        return achieved_goal >= desired_goal


class PickAndPlaceHindsightWrapper(HindsightWrapper):
    def __init__(self, env, geofence):
        super().__init__(env)
        self.pap_env = unwrap_env(env, lambda e: isinstance(e, PickAndPlaceEnv))
        self._geofence = geofence
        self.observation_space = Box(
            low=vectorize(
                Observation(
                    observation=env.observation_space.low,
                    desired_goal=Goal(self.goal_space.low, self.goal_space.low),
                    achieved_goal=None)),
            high=vectorize(
                Observation(
                    observation=env.observation_space.high,
                    desired_goal=Goal(self.goal_space.high, self.goal_space.high),
                    achieved_goal=None)))

    @property
    def goal_space(self):
        return Box(low=np.array([-.14, -.2240, .4]), high=np.array([.11, .2241, .921]))

    def _is_success(self, achieved_goal, desired_goal):
        achieved_goal = Goal(*achieved_goal)
        desired_goal = Goal(*desired_goal)
        block_distance = distance_between(achieved_goal.block, desired_goal.block)
        gripper_distance = distance_between(achieved_goal.gripper, desired_goal.gripper)
        return np.logical_and(block_distance < self._geofence,
                              gripper_distance < self._geofence)

    def _achieved_goal(self):
        return Goal(gripper=self.pap_env.gripper_pos(), block=self.pap_env.block_pos())

    def _desired_goal(self):
        assert isinstance(self.pap_env, PickAndPlaceEnv)
        goal = self.pap_env.initial_block_pos.copy()
        goal[2] += self.pap_env.min_lift_height
        return Goal(gripper=goal, block=goal)


class MultiTaskHindsightWrapper(PickAndPlaceHindsightWrapper):
    def __init__(self, env, geofence):
        self.multi_task_env = unwrap_env(env, lambda e: isinstance(e, MultiTaskEnv))
        super().__init__(env, geofence)
        # tack on gripper goal_space
        self.observation_space = Box(
            low=vectorize([env.observation_space.low, self.goal_space.low]),
            high=vectorize([env.observation_space.high, self.goal_space.high]))

    def _desired_goal(self):
        assert isinstance(self.multi_task_env, MultiTaskEnv)
        goal = self.multi_task_env.goal
        return Goal(goal, goal)

    def step(self, action):
        o2, r, t, info = self.env.step(action)
        new_o2 = Observation(
            observation=o2.observation,
            desired_goal=self._desired_goal(),
            achieved_goal=self._achieved_goal())
        return new_o2, r, t, info

    def reset(self):
        return Observation(
            observation=self.env.reset().observation,
            desired_goal=self._desired_goal(),
            achieved_goal=self._achieved_goal())
