from pathlib import Path

from gym import spaces
import numpy as np

from environments.pick_and_place import PickAndPlaceEnv, Goal


class MultiTaskEnv(PickAndPlaceEnv):
    def __init__(self, steps_per_action, geofence, min_lift_height, render_freq):
        self._goal = None
        super().__init__(fixed_block=False,
                         steps_per_action=steps_per_action,
                         geofence=geofence,
                         min_lift_height=min_lift_height,
                         xml_filepath=Path('models', 'multi-task', 'world.xml'),
                         render_freq=render_freq)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, 1, -1.5, -1.3, -.02]),
            high=np.array([1, 1, 4, .1, 2.3, .4]),
            dtype=np.float32)
        self.lift_height = self._initial_block_pos[2] + geofence + min_lift_height
        self.goal_space = spaces.Box(low=np.array([-.14, -.22, .45]),
                                     high=np.array([.11, .22, .63]))
        self.goals = [np.linspace(start, stop, num) for start, stop, num in
                      zip(self.goal_space.low, self.goal_space.high, [6, 8, 6])]

    def _set_new_goal(self):
        self._goal = np.array([np.random.choice(x) for x in self.goals])

    def set_goal(self, goal):
        self._goal = np.array(goal)

    def goal(self):
        return Goal(gripper=self._goal, block=self._goal)

    def goal_3d(self):
        return self._goal

    def reset_qpos(self):
        self.init_qpos[[self.sim.jnt_qposadr('slide_x'),
                        self.sim.jnt_qposadr('slide_y'),
                        self.sim.jnt_qposadr('arm_flex_joint'),
                        self.sim.jnt_qposadr('wrist_roll_joint'),
                        self.sim.jnt_qposadr('hand_l_proximal_joint'),
                        ]] = np.random.uniform(low=[-.13, -.23, -1.1, -1.575, -0.0083],
                                               high=[.11, .25, .001, 1.575, 0.357])
        r = self.sim.jnt_qposadr('hand_r_proximal_joint')
        l = self.sim.jnt_qposadr('hand_l_proximal_joint')
        self.init_qpos[r] = self.init_qpos[l]

        block_joint = self.sim.jnt_qposadr('block1joint')
        self.init_qpos[[block_joint + 0,
                        block_joint + 1,
                        block_joint + 3,
                        block_joint + 6]] = np.random.uniform(low=list(self.goal_space.low)[:2] + [0, -1],
                                                              high=list(self.goal_space.high)[:2] + [1, 1])
        return self.init_qpos
