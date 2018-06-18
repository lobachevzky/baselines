import abc
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import MultiRNNCell, BasicLSTMCell, LSTMStateTuple

from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input


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
        X, processed_x = observation_input(ob_space, nbatch)
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
        X, processed_x = observation_input(ob_space, nbatch)

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
        X, processed_x = observation_input(ob_space, nbatch)
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
            X, processed_x = observation_input(ob_space, nbatch)
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
                 n_hidden, n_layers, activation, n_lp_layers=0, n_lp_hidden=0, lp_activation=None,
                 reuse=False):  # pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model2", reuse=reuse):
            X, pi_h = observation_input(ob_space, n_batch)
            pi_h = vf_h = lp_h = tf.layers.flatten(pi_h)
            for i in range(n_layers):
                pi_h = activation(fc(pi_h, f'pi_fc{i + 1}', nh=n_hidden, init_scale=np.sqrt(2)))
                vf_h = activation(fc(vf_h, f'vf_fc{i + 1}', nh=n_hidden, init_scale=np.sqrt(2)))
            for i in range(n_lp_layers):
                lp_h = lp_activation(fc(lp_h, f'lp_fc{i + 1}', nh=n_lp_hidden, init_scale=np.sqrt(2)))
            vf = fc(vf_h, 'vf', 1)[:, 0]
            lp = fc(lp_h, 'lp', 1)[:, 0]

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
        self.lp = lp
        self.step = step
        self.value = value


class MlpPolicyLSTMPred:
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_hidden, n_layers,
                 activation, n_cells, n_lstm=256, reuse=False):
        self.pdtype = make_pdtype(ac_space)
        n_batch = n_env * n_steps
        X, pi_h = observation_input(ob_space, n_batch)

        M = tf.placeholder(tf.float32, [n_batch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [n_env, n_lstm * 2])  # states
        with tf.variable_scope("model", reuse=reuse):
            xs = batch_to_seq(X, n_env, n_steps)
            ms = batch_to_seq(M, n_env, n_steps)
            pi_h = vf_h = tf.layers.flatten(pi_h)
            for i in range(n_layers):
                pi_h = activation(fc(pi_h, f'pi_fc{i + 1}', nh=n_hidden, init_scale=np.sqrt(2)))
                vf_h = activation(fc(vf_h, f'vf_fc{i + 1}', nh=n_hidden, init_scale=np.sqrt(2)))
            pi_h = vf_h = tf.layers.flatten(pi_h)
            vf = fc(vf_h, 'vf', 1)[:, 0]
            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h)

        with tf.variable_scope("self_model", reuse=reuse):
            h, s_new = lstm(xs, ms, S, f'lstm{i+1}', nh=n_lstm)
            h5 = seq_to_batch(h)
            lp = fc(h5, 'v', 1)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((n_env, n_lstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, vf, s_new, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(vf, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.lp = lp
        self.step = step
        self.value = value

