#!/usr/bin/env python3
# stdlib
# third party
import argparse
import multiprocessing
import sys
from typing import Iterable

from gym.wrappers import TimeLimit
import numpy as np
import tensorflow as tf

# first party
from baselines import logger
from baselines.bench.monitor import Monitor
from baselines.common.misc_util import set_global_seeds
from baselines.common.models import mlp
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.defaults import mujoco
from baselines.ppo2.hsr_wrapper import (
    HSREnv, Observation, UnsupervisedDummyVecEnv, UnsupervisedEnv,
    UnsupervisedSubprocVecEnv, MoveGripperEnv)
from scripts.hsr import ACTIVATIONS, add_env_args, add_wrapper_args, env_wrapper, parse_activation, parse_groups


def parse_lr(string: str) -> callable:
    return lambda f: float(string) * f


class RewardStructure:
    def __init__(
            self,
            nenv: int,
            params: np.array,
            subspace_sizes: Iterable,
    ):
        self.subspace_sizes = subspace_sizes
        self.trained = False
        param_shape = (Observation(*subspace_sizes).params, )
        assert np.shape(params) == param_shape
        with tf.variable_scope('reward'):
            self.params = tf.get_variable(
                'params',
                (nenv, ) + param_shape,
                initializer=tf.constant_initializer(
                    np.tile(params, (nenv, 1))),
            )

    def recover_observation(self, X: tf.Tensor) -> Observation:
        return Observation(*tf.split(X, self.subspace_sizes, axis=1))

    def function(self, X: tf.Tensor) -> tf.Tensor:
        o = self.recover_observation(X)
        return -tf.reduce_sum(tf.square(o.achieved - o.params))

    def replace_params(self, X: tf.Tensor, params: tf.Tensor) -> tf.Tensor:
        # TODO: Check that params variable matches observation.params
        obs = self.recover_observation(X)
        params = tf.Print(params, [params], message='variable params')
        params = tf.Print(params, [obs.params], message='ph params')
        return tf.concat(obs.replace(params=params), axis=1)


@env_wrapper
def main(max_steps, seed, logdir, env, ncpu, goal_lr, env_args, network_args,
         **kwargs):
    format_strs = ['stdout']
    if logdir:
        format_strs += ['tensorboard']
    logger.configure(format_strs=format_strs, dir=logdir)
    ncpu = nenv = ncpu or multiprocessing.cpu_count()
    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=ncpu,
        inter_op_parallelism_threads=ncpu)
    sess = tf.Session(config=config)
    sess.__enter__()

    def make_env():
        _env = Monitor(
            TimeLimit(max_episode_steps=max_steps, env=env(**env_args)),
            logger.get_dir(),
            allow_early_resets=True)
        _env.seed(seed)
        return _env

    if env is UnsupervisedEnv:
        sample_env = env(**env_args)
        assert isinstance(sample_env, UnsupervisedEnv)
        reward_structure = RewardStructure(
            subspace_sizes=sample_env.subspace_sizes,
            params=sample_env.reward_params,
            nenv=nenv,
        )
        if sys.platform == 'darwin':
            env = UnsupervisedDummyVecEnv(
                [make_env] * nenv, reward_params=reward_structure.params)
        else:
            #TODO: switch to subprocvecenv
            env = UnsupervisedDummyVecEnv(
                [make_env] * nenv, reward_params=reward_structure.params)

        # TODO: make sure tf reward_function and np reward_function are doing the same
        # thing
        # TODO: Is first reset sending wrong params?
        # TODO: What's the deal with eval env? Can we use that to properly
        # evaluate?
    else:
        reward_structure = None
        if sys.platform == 'darwin':
            env = DummyVecEnv([make_env] * nenv)
        else:
            env = SubprocVecEnv([make_env] * nenv)

    # env = VecNormalize(env)

    set_global_seeds(seed)

    model = ppo2.learn(
        reward_structure=reward_structure,
        network='mlp',
        env=env,
        nsteps=max_steps,
        total_timesteps=1e20,
        eval_env=None,
        **kwargs)

    # Run trained model
    logger.log("Running trained model")
    obs = np.zeros((1, ) + env.observation_space.shape)
    obs[:] = env.reset()
    while True:
        actions = model.step(obs)[0]
        obs[:] = env.step(actions)[0]
        env.render()


ENVIRONMENTS = dict(
    move_block=HSREnv,
    move_gripper=MoveGripperEnv,
    unsupervised=UnsupervisedEnv,
)


def add_network_args(parser):
    parser.add_argument('--num-layers', type=int, required=True)
    parser.add_argument('--num-hidden', type=int, required=True)
    parser.add_argument(
        '--activation',
        type=parse_activation,
        default=tf.nn.relu,
        choices=ACTIVATIONS.values())


def cli():
    parser = argparse.ArgumentParser()

    add_wrapper_args(parser=parser.add_argument_group('wrapper_args'))
    add_env_args(parser=parser.add_argument_group('env_args'))
    add_network_args(parser=parser.add_argument_group('network_args'))

    parser.add_argument(
        '--env',
        choices=ENVIRONMENTS.values(),
        type=lambda k: ENVIRONMENTS[k],
        default=HSREnv)
    parser.add_argument('--max-steps', type=int, required=True)
    parser.add_argument('--ncpu', type=int)
    parser.add_argument('--goal-lr', type=eval)
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--max-grad-norm', type=float, required=True)
    parser.add_argument('--nminibatches', type=int)
    parser.add_argument('--lam', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--noptepochs', type=int)
    parser.add_argument('--log-interval', type=int)
    parser.add_argument('--ent-coef', type=float)
    parser.add_argument('--lr', type=parse_lr)
    parser.add_argument('--cliprange', type=float)
    parser.add_argument('--value-network')
    parser.add_argument('--normalize-observations', action='store_true')
    parser.set_defaults(**mujoco())

    main(**(parse_groups(parser)))


if __name__ == '__main__':
    cli()
