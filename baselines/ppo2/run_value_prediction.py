#!/usr/bin/env python3
import click
import numpy as np
import tensorflow as tf
from gym.wrappers import TimeLimit

from baselines import bench, logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy
from environments.pick_and_place import PickAndPlaceEnv


@click.command()
@click.option('--seed', default=1, type=int)
@click.option('--max-steps', default=300, type=int)
@click.option('--steps-per-action', default=200, type=int)
@click.option('--fixed-block', is_flag=True)
@click.option('--min-lift-height', default=.02, type=float)
@click.option('--geofence', default=.4, type=float)
@click.option('--max-grad-norm', default=.5, type=float)
@click.option('--n-mini-batch', default=32, type=int)
@click.option('--n-layers', default=2, type=int)
@click.option('--n-hidden', default=128, type=int)
@click.option('--n-steps', default=2048, type=int)
@click.option('--tanh', 'activation', flag_value=tf.nn.tanh)
@click.option('--relu', 'activation', flag_value=tf.nn.relu, default=True)
@click.option('--logdir', type=str)
def cli(max_steps, steps_per_action, fixed_block, min_lift_height, geofence, seed,
        logdir, n_mini_batch, n_steps, n_layers, n_hidden, activation, max_grad_norm):
    format_strs = ['stdout']
    if logdir:
        format_strs += ['tensorboard']
    logger.configure(format_strs=format_strs, dir=logdir)
    env = TimeLimit(
        max_episode_steps=max_steps,
        env=PickAndPlaceEnv(
            discrete=False,
            cheat_prob=0,
            steps_per_action=steps_per_action,
            fixed_block=fixed_block,
            min_lift_height=min_lift_height,
            geofence=geofence,
            xml_file='world.xml',
            render_freq=0,
        ))
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
        return MlpPolicy(
            n_hidden=128,
            n_layers=2,
            activation=tf.nn.relu,
            n_lp_layers=n_layers,
            n_lp_hidden=n_hidden,
            lp_activation=activation,
            *args, **kwargs)

    model = ppo2.learn(policy=policy, env=env, n_steps=n_steps, n_mini_batches=n_mini_batch,
                       max_grad_norm=max_grad_norm,
                       lam=0.95, gamma=0.99, n_opt_epochs=10, log_interval=1,
                       ent_coef=0.0,
                       lr=3e-4,
                       clip_range=0.2,
                       total_time_steps=1e20,
                       predict_loss=True)

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
