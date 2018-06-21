#!/usr/bin/env python3

import click
import numpy as np
import tensorflow as tf
from gym.wrappers import TimeLimit

from baselines import bench, logger
from baselines.common import set_global_seeds, gym
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.networks import MlpPolicy, MlpPolicyWithMemory
from environments.frozen_lake_goal import FrozenLakeGoalEnv


@click.command()
@click.option('--seed', default=1, type=int)
@click.option('--max-steps', default=300, type=int)
@click.option('--max-grad-norm', default=.5, type=float)
@click.option('--n-mini-batch', default=1, type=int)
@click.option('--n-memory', default=16, type=int)
@click.option('--n-env', default=1, type=int)
@click.option('--size-memory', default=128, type=int)
@click.option('--map-dim', default=8, type=int)
@click.option('--n-steps', default=300, type=int)
@click.option('--mlp', 'network', flag_value='mlp')
@click.option('--mem', 'network', flag_value='mem', default=True)
@click.option('--tanh', 'activation', flag_value=tf.nn.tanh)
@click.option('--relu', 'activation', flag_value=tf.nn.relu, default=True)
@click.option('--logdir', type=str)
def cli(max_steps, seed, network, logdir, n_mini_batch, n_steps,
        n_memory, size_memory, activation, max_grad_norm, n_env, map_dim):
    format_strs = ['stdout']
    if logdir:
        format_strs += ['tensorboard']
    logger.configure(format_strs=format_strs, dir=logdir)

    env = TimeLimit(max_episode_steps=max_steps,
                    env=FrozenLakeGoalEnv(map_dims=(map_dim, map_dim)))
    # env = gym.make('FrozenLake-v0')
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    def make_env():
        return bench.Monitor(env, logger.get_dir(), allow_early_resets=True)

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)

    def policy(*args, **kwargs):
        if network == 'mlp':
            return MlpPolicy(
                n_hidden=128,
                n_layers=2,
                activation=activation,
                *args, **kwargs)
        else:
            return MlpPolicyWithMemory(
                size_layer=128,
                n_layers=2,
                activation=activation,
                n_memory=n_memory,
                size_memory=size_memory,
                *args, **kwargs)

    model = ppo2.learn(policy=policy, env=env, n_steps=n_steps, n_mini_batches=n_mini_batch,
                       max_grad_norm=max_grad_norm,
                       lam=0.95, gamma=0.99, n_opt_epochs=10, log_interval=1,
                       ent_coef=0.0,
                       lr=3e-4,
                       clip_range=0.2,
                       total_time_steps=1e20,
                       remember_states=False)

    # Run trained model
    logger.log("Running trained model")
    obs = np.zeros((env.num_envs,) + env.observation_space.shape)
    obs[:] = env.reset()
    while True:
        actions = model.step(obs)[0]
        obs[:] = env.step(actions)[0]
        env.render()


if __name__ == '__main__':
    cli()