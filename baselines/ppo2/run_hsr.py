#!/usr/bin/env python3
# stdlib
import argparse
import multiprocessing

# third party
from environments.hsr import Observation
from gym.wrappers import TimeLimit
import numpy as np
from scripts.hsr import ACTIVATIONS, add_env_args, add_wrapper_args, env_wrapper, parse_activation, parse_groups
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
from baselines.ppo2.hsr_wrapper import HSREnv, UnsupervisedDummyVecEnv, UnsupervisedEnv, UnsupervisedVecEnv


def parse_lr(string: str) -> callable:
    return lambda f: float(string) * f


class RewardStructure:
    def __init__(self, subspace_sizes):
        self.subspace_sizes = subspace_sizes
        param_shape = [Observation(*subspace_sizes).goal]
        with tf.variable_scope('reward'):
            self.params = tf.get_variable('params', shape=param_shape)

    def function(self, X: tf.Tensor):
        achieved = Observation(*tf.split(X, self.subspace_sizes, axis=1)).goal
        return -tf.reduce_sum(tf.square(achieved - self.params))


@env_wrapper
def main(max_steps, seed, logdir, env, ncpu, goal_lr, env_args, network_args,
         **kwargs):
    format_strs = ['stdout']
    if logdir:
        format_strs += ['tensorboard']
    logger.configure(format_strs=format_strs, dir=logdir)
    ncpu = ncpu or multiprocessing.cpu_count()
    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=ncpu,
        inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    sample_env = env(**env_args)

    def make_env():
        _env = Monitor(
            TimeLimit(max_episode_steps=max_steps, env=env(**env_args)),
            logger.get_dir(),
            allow_early_resets=True)
        _env.seed(seed)
        return _env

    if env is UnsupervisedEnv:
        assert isinstance(sample_env, UnsupervisedEnv)
        reward_structure = RewardStructure(
            subspace_sizes=sample_env.subspace_sizes)
        # env = UnsupervisedVecEnv([make_env for _ in range(ncpu)],
        #                          reward_params=reward_structure.params)
        env = UnsupervisedDummyVecEnv([make_env],
                                      reward_params=reward_structure.params)

        def network(X: tf.Tensor):
            nbatch = tf.shape(X)[0]
            reward_params = tf.tile(
                tf.expand_dims(reward_structure.params, axis=0), [nbatch, 1])
            inputs = tf.concat([X, reward_params], axis=1)
            return mlp(**network_args)(inputs)

        # TODO: make sure tf reward_function and np reward_function are doing the same
        # thing
    else:
        reward_structure = None
        # env = SubprocVecEnv([make_env for _ in range(ncpu)])
        env = DummyVecEnv([make_env])
        network = 'mlp'

    env = VecNormalize(env)

    set_global_seeds(seed)

    model = ppo2.learn(
        reward_structure=reward_structure,
        network=network,
        env=env,
        total_timesteps=1e20,
        eval_env=env,
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
    parser.add_argument('--nsteps', type=int)
    parser.add_argument('--nminibatches', type=int)
    parser.add_argument('--lam', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--noptepochs', type=int)
    parser.add_argument('--log-interval', type=int)
    parser.add_argument('--ent-coef', type=float)
    parser.add_argument('--lr', type=parse_lr)
    parser.add_argument('--cliprange', type=float)
    parser.add_argument('--value-network')
    parser.set_defaults(**mujoco())

    main(**(parse_groups(parser)))


if __name__ == '__main__':
    cli()
