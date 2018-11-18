#!/usr/bin/env python3
import argparse
import multiprocessing

from gym.wrappers import TimeLimit
import numpy as np
import tensorflow as tf

from baselines import bench, logger
from baselines.common.misc_util import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.defaults import mujoco
from baselines.ppo2.hsr_wrapper import HSREnv, UnsupervisedEnv, UnsupervisedVecEnv
from scripts.hsr import ACTIVATIONS, add_env_args, add_wrapper_args, env_wrapper, parse_activation, parse_groups


def parse_lr(string: str) -> callable:
    return lambda f: float(string) * f


@env_wrapper
def main(max_steps, seed, logdir, env, ncpu, goal_lr, goal_activation,
         goal_n_layers, goal_layer_size, env_args, **kwargs):

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

    def make_env(_seed):
        _env = bench.Monitor(
            TimeLimit(
                max_episode_steps=max_steps, env=env(seed=_seed, **env_args)),
            logger.get_dir(),
            allow_early_resets=True)
        _env.seed(_seed)
        return _env

    if env == 'unsupervised':
        reward_function = UnsupervisedEnv.reward_function
        env = UnsupervisedVecEnv([make_env(i + seed) for i in range(ncpu)])
    else:
        reward_function = None
        env = DummyVecEnv([make_env])

    env = VecNormalize(env)

    set_global_seeds(seed)

    model = ppo2.learn(
        reward_function=reward_function,
        network='mlp',
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


def cli():
    parser = argparse.ArgumentParser()

    add_wrapper_args(parser=parser.add_argument_group('wrapper_args'))
    add_env_args(parser=parser.add_argument_group('env_args'))

    parser.add_argument(
        '--env',
        choices=ENVIRONMENTS.values(),
        type=lambda k: ENVIRONMENTS[k],
        default=HSREnv)
    parser.add_argument('--max-steps', type=int, required=True)
    parser.add_argument('--ncpu', type=int)
    parser.add_argument(
        '--goal-activation',
        type=parse_activation,
        default=tf.nn.relu,
        choices=ACTIVATIONS.values())
    parser.add_argument('--goal-n-layers', type=int)
    parser.add_argument('--goal-layer-size', type=int)
    parser.add_argument('--goal-lr', type=eval)
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--max-grad-norm', type=float, required=True)
    parser.add_argument('--num-layers', type=int, required=True)
    parser.add_argument('--num-hidden', type=int, required=True)
    parser.add_argument(
        '--activation',
        type=parse_activation,
        default=tf.nn.relu,
        choices=ACTIVATIONS.values())
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
