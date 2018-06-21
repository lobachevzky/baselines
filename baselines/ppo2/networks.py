import abc
from contextlib import contextmanager
from typing import List

import numpy as np
import tensorflow as tf
from gym import spaces
from tensorflow.contrib.rnn import LSTMBlockFusedCell
from tensorflow.python.ops.rnn_cell_impl import MultiRNNCell, BasicLSTMCell, LSTMStateTuple

from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, ortho_init
from baselines.common.distributions import make_pdtype
from baselines.common.input import get_inputs


def nature_cnn(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))


class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        X, processed_x = get_inputs(ob_space, nbatch)
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(processed_x)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value


class LstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = get_inputs(ob_space, nbatch)

        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value


class CnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **conv_kwargs):  # pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = get_inputs(ob_space, nbatch)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(processed_x, **conv_kwargs)
            vf = fc(h, 'v', 1)[:, 0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value


class MlpPolicyOld(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):  # pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("old_model", reuse=reuse):
            X, processed_x = get_inputs(ob_space, nbatch)
            activ = tf.tanh
            processed_x = tf.layers.flatten(processed_x)
            pi_h1 = activ(fc(processed_x, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(processed_x, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:, 0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value


class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, n_batch, n_steps,
                 n_hidden, n_layers, activation, reuse=False):  # pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X, pi_h = get_inputs(ob_space, n_batch)
            pi_h = vf_h = lp_h = tf.layers.flatten(pi_h)
            for i in range(n_layers):
                pi_h = activation(fc(pi_h, f'pi_fc{i + 1}', nh=n_hidden, init_scale=np.sqrt(2)))
                vf_h = activation(fc(vf_h, f'vf_fc{i + 1}', nh=n_hidden, init_scale=np.sqrt(2)))
            vf = fc(vf_h, 'vf', 1)[:, 0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value


@contextmanager
def check_shape(tensor, shape):
    tf_shape = tf.shape(tensor)
    with tf.control_dependencies([tf.Assert(tf_shape == shape, [tf_shape])]):
        yield


class MlpPolicyWithMemory:
    # noinspection PyPep8Naming
    def __init__(self, sess, ob_space, ac_space, n_batch, n_steps,
                 size_layer, n_layers, activation, n_memory, size_memory,
                 reuse=False):  # pylint: disable=W0613
        def dense(x, scope: str, nh=size_layer, reuse: bool = False):
            return activation(fc(x, scope=scope, nh=nh,
                                 init_scale=np.sqrt(2), reuse=reuse))

        self.pdtype = make_pdtype(ac_space)
        assert isinstance(ob_space, spaces.Dict)
        input_dict = get_inputs(ob_space, n_batch)
        G, goal_h = input_dict['goal']
        ob_shape = [n_batch] + list(ob_space.spaces['obs'].shape)
        goal_shape = [n_batch] + list(ob_space.spaces['goal'].shape)
        with check_shape(goal_h, ob_shape):
            O, obs_h = input_dict['obs']
        with check_shape(obs_h, ob_shape):
            A, action_h = get_inputs(ac_space, n_batch)
        with check_shape(action_h, [n_batch] + list(ac_space.shape)):
            mem_shape = [n_batch, n_memory, size_memory]
            self.initial_state = tf.orthogonal_initializer()(mem_shape, tf.float32)
            M = tf.placeholder(tf.float32, mem_shape)

        def c(h):
            with check_shape(h, [n_batch, size_memory]):
                key = tf.expand_dims(h, axis=1)
            with check_shape(key, [n_batch, 1, size_memory]):
                prod = key * M
            with check_shape(prod, mem_shape):
                sim = tf.reduce_sum(prod, axis=1, keepdims=True)
            with check_shape(sim, [n_batch, 1, size_memory]):
                return tf.nn.softmax(sim, axis=1)

        with tf.variable_scope("network", reuse=reuse):
            for i in range(n_layers):
                obs_h = dense(obs_h, f'obs_fc{i + 1}')
                with check_shape(obs_h, [n_batch, size_layer]):
                    action_h = dense(action_h, f'action_fc{i + 1}')
                with check_shape(action_h, [n_batch, size_layer]):
                    goal_h = dense(goal_h, f'obs_fc{i + 1}', reuse=True)
            with check_shape(goal_h, [n_batch, size_layer]):
                h = tf.concat([goal_h, obs_h], axis=1)
            with check_shape(h, [n_batch, 2 * size_layer]):
                vf = fc(h, 'vf', 1)[:, 0]
                self.pd, self.pi = self.pdtype.pdfromlatent(h)

            oa_h = dense(obs_h, 'obs_fc_final', nh=size_memory) + \
                   dense(action_h, 'action_fc_final', nh=size_memory)
            with check_shape(oa_h, [n_batch, size_memory]):
                oa_h -= tf.reduce_mean(oa_h, axis=1, keepdims=True)
            with check_shape(oa_h, [n_batch, size_memory]):
                goal_h = dense(goal_h, 'goal_fc_final')
            c_oa = c(oa_h)
            with check_shape(c_oa, [n_batch, size_memory]):
                c_g = c(goal_h)
            with check_shape(c_oa, [n_batch, size_memory]):
                M_new = M + (tf.expand_dims(oa_h, axis=1) *
                             tf.expand_dims(c_oa, axis=2))

            c_oa = tf.distributions.Categorical(c_oa)
            c_g = tf.distributions.Categorical(c_g)
            divergence = tf.distributions.kl_divergence(c_oa, c_g)
            with check_shape(divergence, [n_batch, 1]):
                qf = 1 / divergence
                qf /= 1 + qf

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)

        def ob_feed(ob):
            return {self.X[k]: ob[k] for k in ob}

        def step(ob, memory, _):
            return sess.run([a0, vf, M_new, neglogp0], {**{ob_feed(ob)}, M: memory})

        def value(ob, memory, _):
            return sess.run(vf, {**{ob_feed(ob)}, M: memory})

        self.X = dict(goal=G, obs=O)
        self.M = M
        self.S = None
        self.vf = vf
        self.qf = qf
        self.step = step
        self.value = value
