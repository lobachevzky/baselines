#!/usr/bin/env python3
import argparse
import multiprocessing

import numpy as np
import tensorflow as tf
from gym.wrappers import TimeLimit
from scripts.hsr import env_wrapper, ACTIVATIONS, parse_activation, \
    add_wrapper_args, add_env_args, parse_groups

from baselines import bench, logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.hsr_wrapper import HSREnv, MoveGripperEnv


@env_wrapper
def main(
        max_steps,
        seed,
        grad_clip,
        nminibatches,
        logdir,
        env,
        nsteps,
        ncpu,
        activation,
        num_layers,
        num_hidden,
        learning_rate,
        goal_learning_rate,
        goal_activation,
        goal_n_layers,
        goal_layer_size,
        env_args):

    format_strs = ['stdout']
    if logdir:
        format_strs += ['tensorboard']
    logger.configure(format_strs=format_strs, dir=logdir)
    ncpu = ncpu or multiprocessing.cpu_count()
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    def make_env():
        return bench.Monitor(
            TimeLimit(max_episode_steps=max_steps, env=env(**env_args)),
            logger.get_dir(), allow_early_resets=True)

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)

    model = ppo2.learn(network='mlp', env=env,
                       total_timesteps=1e20,
                       eval_env=env,
                       seed=seed,
                       nsteps=nsteps,
                       max_grad_norm=grad_clip,
                       nminibatches=nminibatches,
                       lam=0.95, gamma=0.99, log_interval=1,
                       ent_coef=0.0,
                       lr=learning_rate,
                       num_layers=num_layers,
                       num_hidden=num_hidden,
                       activation=activation)

    # Run trained model
    logger.log("Running trained model")
    obs = np.zeros((1,) + env.observation_space.shape)
    obs[:] = env.reset()
    while True:
        actions = model.step(obs)[0]
        obs[:] = env.step(actions)[0]
        env.render()


ENVIRONMENTS = dict(
    move_block=HSREnv,
    move_gripper=MoveGripperEnv,
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
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument(
        '--activation',
        type=parse_activation,
        default=tf.nn.relu,
        choices=ACTIVATIONS.values())
    parser.add_argument('--num-layers', type=int, required=True)
    parser.add_argument('--num-hidden', type=int, required=True)
    parser.add_argument(
        '--goal-activation',
        type=parse_activation,
        default=tf.nn.relu,
        choices=ACTIVATIONS.values())
    parser.add_argument('--goal-n-layers', type=int)
    parser.add_argument('--goal-layer-size', type=int)
    parser.add_argument('--goal-learning-rate', type=float)
    parser.add_argument('--nminibatches', type=int, required=True)
    parser.add_argument('--nsteps', type=int, required=True)
    parser.add_argument('--learning-rate', type=eval, required=True)
    parser.add_argument('--grad-clip', type=float, required=True)
    parser.add_argument('--logdir', type=str, default=None)

    main(**(parse_groups(parser)))


if __name__ == '__main__':
    cli()
