import itertools
from collections import namedtuple

import numpy as np
from gym import spaces

from baselines.her.util import vectorize
from environments.mujoco import distance_between
from environments.pick_and_place import PickAndPlaceEnv
from mujoco import ObjType

Observation = namedtuple('Obs', 'observation goal')

Goal = namedtuple('Goal', 'gripper block')

class MultiTaskEnv(PickAndPlaceEnv):
    def __init__(self, geofence: float, randomize_pose=False,
                 fixed_block=False, fixed_goal=None, **kwargs):
        self.fixed_block = fixed_block
        self.fixed_goal = fixed_goal
        self.randomize_pose = randomize_pose
        self.geofence = geofence
        self.goal_space = spaces.Box(
            low=np.array([-.13, -.21, .40]), high=np.array([.10, .21, .4001]))
        self.goal = self.goal_space.sample() if fixed_goal is None else fixed_goal
        super().__init__(fixed_block=False, **kwargs)
        # low=np.array([-.14, -.22, .40]), high=np.array([.11, .22, .63]))

        goal_size = np.array([.0317, .0635, .0234]) * geofence
        x, y, z = [
            np.arange(l, h, s)
            for l, h, s in zip(self.goal_space.low, self.goal_space.high, goal_size)
        ]
        # goal_corners = np.array(list(itertools.product(x, y, z)))
        # self.labels = {tuple(g): '.' for g in goal_corners}

    def _is_successful(self):
        return distance_between(self.goal, self.block_pos()) < self.geofence

    def _get_obs(self):
        desired_goal = vectorize([self.goal, self.goal])
        achieved_goal = vectorize([self.gripper_pos(), self.block_pos()])
        return dict(
            observation=super()._get_obs(),
            desired_goal=desired_goal,
            achieved_goal=achieved_goal
        )

    def _reset_qpos(self):
        if self.randomize_pose:
            for joint in [
                'slide_x', 'slide_y', 'arm_lift_joint', 'arm_flex_joint',
                'wrist_roll_joint', 'hand_l_proximal_joint'
            ]:
                qpos_idx = self.sim.get_jnt_qposadr(joint)
                jnt_range_idx = self.sim.name2id(ObjType.JOINT, joint)
                self.init_qpos[qpos_idx] = np.random.uniform(
                    *self.sim.jnt_range[jnt_range_idx])

        right = self.sim.get_jnt_qposadr('hand_r_proximal_joint')
        left = self.sim.get_jnt_qposadr('hand_l_proximal_joint')
        self.init_qpos[right] = self.init_qpos[left]

        block_joint = self.sim.get_jnt_qposadr('block1joint')
        if not self.fixed_block:
            self.init_qpos[[
                block_joint + 0, block_joint + 1, block_joint + 3, block_joint + 6
            ]] = np.random.uniform(
                low=list(self.goal_space.low)[:2] + [0, -1],
                high=list(self.goal_space.high)[:2] + [1, 1])
        return self.init_qpos

    def reset(self):
        if self.fixed_goal is None:
            self.goal = self.goal_space.sample()
        return super().reset()

    def render(self, labels=None, **kwargs):
        if labels is None:
            labels = dict()
        labels[tuple(self.goal)] = 'x'
        return super().render(labels=labels, **kwargs)

    def step(self, action):
        s, r, t, i = super().step(action)
        i['is_success'] = self._is_successful()
        return s, r, t, i

    def _is_success(self, achieved_goal, desired_goal):
        achieved_goal = Goal(*achieved_goal)
        desired_goal = Goal(*desired_goal)
        block_distance = distance_between(achieved_goal.block, desired_goal.block)
        gripper_distance = distance_between(achieved_goal.gripper, desired_goal.gripper)
        return np.logical_and(block_distance < self._geofence,
                              gripper_distance < self._geofence)

    def _desired_goal(self):
        goal = self.goal
        return Goal(goal, goal)

    def _achieved_goal(self):
        return Goal(gripper=self.gripper_pos(), block=self.block_pos())


    def compute_reward(self, achieved_goal=None, desired_goal=None, info=None):
        if achieved_goal is None:
            achieved_goal = self._achieved_goal()
        if desired_goal is None:
            desired_goal = self._desired_goal()
        if info is None:
            info = {}
        if isinstance(achieved_goal, np.ndarray):
            achieved_goal = Goal(*np.split(achieved_goal, 2, axis=-1))
        if isinstance(desired_goal, np.ndarray):
            desired_goal = Goal(*np.split(desired_goal, 2, axis=-1))
        return np.logical_and(distance_between(achieved_goal.gripper, desired_goal.gripper) < self.geofence,
             distance_between(achieved_goal.block, desired_goal.block) < self.geofence)
