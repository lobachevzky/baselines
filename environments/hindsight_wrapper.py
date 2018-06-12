from abc import abstractmethod
from collections import namedtuple
from typing import Iterable, List

import gym
import numpy as np
from gym.spaces import Box

from environments.mujoco import distance_between
from environments.pick_and_place import Goal
from sac.utils import Step


class State(namedtuple('State', 'observation achieved_goal desired_goal')):
    def replace(self, **kwargs):
        return super()._replace(**kwargs)


class HindsightWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        vector_state = self.vectorize_state(self.reset())
        self.observation_space = Box(-1, 1, vector_state.shape[1:])

    @abstractmethod
    def _achieved_goal(self):
        raise NotImplementedError

    @abstractmethod
    def _is_success(self, achieved_goal, desired_goal):
        raise NotImplementedError

    @abstractmethod
    def _desired_goal(self):
        raise NotImplementedError

    @staticmethod
    def vectorize_state(state):
        return np.concatenate(state)

    def step(self, action):
        s2, r, t, info = self.env.step(action)
        new_s2 = State(
            observation=s2,
            desired_goal=self._desired_goal(),
            achieved_goal=self._achieved_goal())
        is_success = self._is_success(new_s2.achieved_goal, new_s2.desired_goal)
        new_t = is_success or t
        new_r = float(is_success)
        info['base_reward'] = r
        return new_s2, new_r, new_t, info

    def reset(self):
        return State(
            observation=self.env.reset(),
            desired_goal=self._desired_goal(),
            achieved_goal=self._achieved_goal())

    def recompute_trajectory(self, trajectory: Iterable, final_step: Step):
        achieved_goal = None
        for step in trajectory:
            if achieved_goal is None:
                achieved_goal = final_step.s2.achieved_goal
            new_t = self._is_success(step.s2.achieved_goal, achieved_goal)
            r = float(new_t)
            yield Step(
                s1=step.s1.replace(desired_goal=achieved_goal),
                a=step.a,
                r=r,
                s2=step.s2.replace(desired_goal=achieved_goal),
                t=new_t)
            if new_t:
                break


class MountaincarHindsightWrapper(HindsightWrapper):
    """
    new obs is [pos, vel, goal_pos]
    """

    def _achieved_goal(self):
        return self.env.unwrapped.state[0]

    def _is_success(self, achieved_goal, desired_goal):
        return self.env.unwrapped.state[0] >= self._desired_goal()

    def _desired_goal(self):
        return 0.45

    @staticmethod
    def vectorize_state(states: List[State]):
        if isinstance(states, State):
            states = [states]
        return np.stack(
            np.append(state.observation, state.desired_goal) for state in states)


class PickAndPlaceHindsightWrapper(HindsightWrapper):
    def __init__(self, env):
        super().__init__(env)

    def _is_success(self, achieved_goal, desired_goal):
        geofence = self.env.unwrapped.geofence
        return distance_between(achieved_goal.block, desired_goal.block) < geofence and \
               distance_between(achieved_goal.gripper,
                                desired_goal.gripper) < geofence

    def _achieved_goal(self):
        return Goal(
            gripper=self.env.unwrapped.gripper_pos(),
            block=self.env.unwrapped.block_pos())

    def _desired_goal(self):
        return self.env.unwrapped.goal()

    @staticmethod
    def vectorize_state(states: List[State]):
        """
        :returns
        >>> np.stack([np.concatenate(
        >>>    [state.observation, state.desired_goal.gripper, state.desired_goal.block])
        >>>     for state in states])
        """
        if isinstance(states, State):
            states = [states]

        def get_arrays(s: State):
            return [s.observation, s.desired_goal.gripper, s.desired_goal.block]

        slices = np.cumsum([0] + [np.size(a) for a in get_arrays(states[0])])
        state_vector = np.empty((len(states), slices[-1]))
        for i, state in enumerate(states):
            for (start, stop), array in zip(zip(slices, slices[1:]), get_arrays(state)):
                state_vector[i, start:stop] = array

        return state_vector


class MultiTaskHindsightWrapper(PickAndPlaceHindsightWrapper):
    def __init__(self, env):
        super().__init__(env)

    def recompute_trajectory(self, reverse_trajectory: Iterable, final_step: Step):
        achieved_goals = []
        last_goal = True
        for step in reverse_trajectory:
            assert isinstance(step, Step)
            achieved_goal = step.s2.achieved_goal

            if last_goal:
                last_goal = False
                if np.random.uniform(0, 1) < .1:
                    achieved_goals.append(achieved_goal)

            block_lifted = achieved_goal.block[2] > self.env.unwrapped.lift_height
            in_box = achieved_goal.block[1] > .1 and not block_lifted
            if block_lifted or in_box:
                achieved_goals.append(achieved_goal)

            for achieved_goal in achieved_goals:
                new_t = self._is_success(achieved_goal=step.s2.achieved_goal,
                                         desired_goal=achieved_goal)
                r = float(new_t)
                yield Step(
                    s1=step.s1.replace(desired_goal=achieved_goal),
                    a=step.a,
                    r=r,
                    s2=step.s2.replace(desired_goal=achieved_goal),
                    t=new_t)
