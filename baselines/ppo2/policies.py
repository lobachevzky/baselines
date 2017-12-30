import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMCell, LSTMStateTuple

from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype


class MemoryPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, size_mem=256, reuse=False):
        nenv = nbatch // nsteps

        # nh, nw, nc = ob_space.shape
        # ob_shape = (nbatch, nh, nw, nc)
        ob_shape = (nbatch,) + ob_space.shape
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, size_mem * 2])  # states
        with tf.variable_scope("model", reuse=reuse):
            h = self.preprocess(X)
            h = fc(h, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = self.memory_fn(xs, ms, S, nh=size_mem)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x: x)
            vf = fc(h5, 'v', 1, act=lambda x: x)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, size_mem * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

    @staticmethod
    def preprocess(X):
        h = conv(tf.cast(X, tf.float32) / 255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
        h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
        h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
        return conv_to_fc(h3)

    @staticmethod
    def memory_fn(xs, ms, S, nh):
        raise NotImplemented


class LnLstmPolicy(MemoryPolicy):
    @staticmethod
    def memory_fn(xs, ms, S, nh):
        return lnlstm(xs, ms, S, 'lnlstm1', nh=nh)

        # class LstmPolicy(MemoryPolicy):
        #     @staticmethod
        #     def memory_fn(xs, ms, S, nh):
        #         return lstm(xs, ms, S, 'lstm1', nh=nh)

        # @staticmethod
        # def preprocess(X):
        #     return tf.cast(X, tf.float32)


def squash(vector, epsilon=1e-9):
    """Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    """
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    # element-wise
    return scalar_factor * vector


def routing(inputs, b_IJ, output_size, stddev=1.0, iter_routing=2):
    """ The routing algorithm.
    Args:
        inputs: A Tensor with [batch_size, num_caps_i=1152, 1, length(u_i)=8, 1]
               shape, num_caps_i meaning the number of capsule in the layer l.
    Returns:
        A Tensor of shape [batch_size, num_caps_j, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     """

    # num_caps_j = 10
    # len_u_i = 8
    # len_v_j = 16
    len_v_j = output_size
    batch_size, num_caps_i, num_caps_j = b_IJ.get_shape()
    len_u_i = inputs.get_shape()[-1]

    print('iter_routing', iter_routing)
    print('n_caps_i', num_caps_i)
    print('n_caps_j', num_caps_j)

    # num_caps_i = 1152
    b_IJ = tf.reshape(b_IJ, [batch_size, num_caps_i, num_caps_j, 1, 1])
    assert b_IJ.get_shape() == [batch_size, num_caps_i, num_caps_j, 1, 1]
    assert inputs.get_shape() == [batch_size, num_caps_i, len_u_i]

    # W: [num_caps_i, num_caps_j, len_u_i, len_v_j]
    W = tf.get_variable('Weight', shape=(1, num_caps_i, num_caps_j, len_u_i, len_v_j), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=stddev))
    # W = tf.get_variable('Weight', shape=(len_u_i, len_v_j), dtype=tf.float32,
    #                     initializer=tf.random_normal_initializer(stddev=stddev))
    assert W.shape == [1, num_caps_i, num_caps_j, len_u_i, output_size]

    # Eq.2, calc u_hat
    # do tiling for input and W before matmul
    # input => [batch_size, 1152, 10, 8, 1]
    # W => [batch_size, 1152, 10, 8, 16]
    inputs = tf.reshape(inputs, [batch_size, num_caps_i, 1, len_u_i, 1])
    inputs = tf.tile(inputs, [1, 1, num_caps_j, 1, 1])
    W = tf.tile(W, [batch_size, 1, 1, 1, 1])
    assert inputs.get_shape() == [batch_size, num_caps_i, num_caps_j, len_u_i, 1]

    # in last 2 dims:
    # [8, 16].T x [8, 1] => [16, 1] => [batch_size, 1152, 10, 16, 1]
    # tf.scan, 3 iter, 1080ti, 128 batch size: 10min/epoch
    # u_hat = tf.scan(lambda ac, x: tf.matmul(W, x, transpose_a=True), input, initializer=tf.zeros([1152, 10, 16, 1]))
    # tf.tile, 3 iter, 1080ti, 128 batch size: 6min/epoch
    u_hat = tf.matmul(W, inputs, transpose_a=True)
    assert u_hat.get_shape() == [batch_size, num_caps_i, num_caps_j, len_v_j, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b_IJ, dim=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                # s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                # assert s_J.get_shape() == [batch_size, 1, num_caps_j, len_v_j, 1]
                #
                # # line 6:
                # # squash using Eq.1,
                # v_J = squash(s_J)
                # assert v_J.get_shape() == [batch_size, 1, num_caps_j, len_v_j, 1]
            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
            s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
            v_J = squash(s_J)

            # line 7:
            # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
            # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
            # batch_size dim, resulting in [1, 1152, 10, 1, 1]
            v_J_tiled = tf.tile(v_J, [1, num_caps_i, 1, 1, 1])
            u_dot_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
            assert u_dot_v.get_shape() == [batch_size, num_caps_i, num_caps_j, 1, 1]

            # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
            b_IJ += u_dot_v

    return v_J


class CapsulesPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, size_mem=256, reuse=False):  # pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        if ac_space.shape == ():
            actdim = 1
        else:
            actdim = ac_space.shape[0]

        nenv = nbatch // nsteps

        X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        M = tf.placeholder(tf.float32, [nbatch], name='M')  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, size_mem * 2], name='S')  # states
        snew = S

        with tf.variable_scope("model", reuse=reuse):
            n_capsules = 2
            size = 64
            h1 = fc(X, 'pi_fc1', nh=n_capsules * size, init_scale=np.sqrt(2), act=tf.tanh)

            b_IJ = tf.zeros([nbatch, n_capsules, n_capsules], dtype=np.float32)
            h4 = tf.reshape(h1, shape=[nbatch, n_capsules, size])
            h5 = routing(inputs=h4, b_IJ=b_IJ, output_size=size)
            assert h5.shape == [nbatch, 1, n_capsules, size, 1]
            h2 = tf.reshape(h5, shape=[nbatch, n_capsules * size])

            # h2 = fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)

            pi = fc(h2, 'pi', actdim, act=lambda x: x, init_scale=0.01)
            h1 = fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            vf = fc(h2, 'vf', 1, act=lambda x: x)[:, 0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                                     initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, size_mem * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, vf, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(vf, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CapsulesPolicy2(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, size_mem=256, reuse=False):  # pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        if ac_space.shape == ():
            actdim = 1
        else:
            actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs

        nenv = nbatch // nsteps
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, size_mem * 2])  # states

        with tf.variable_scope("model", reuse=reuse):
            # h1 = fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            # h2 = fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)

            X_size = np.prod(ob_space.shape)
            h2 = tf.reshape(X, shape=[nbatch, X_size])
            n_capsules = 2
            h3 = fc(h2, 'h3', n_capsules * X_size)
            # xs = batch_to_seq(h2, nenv, nsteps)
            # ms = batch_to_seq(M, nenv, nsteps)
            # h5, snew = lstm(xs, ms, S, 'lstm', nh=size_mem)
            # h5 = seq_to_batch(h5)
            snew = S
            # h5 = fc(h3, 'h5', n_capsules * X_size)

            b_IJ = tf.zeros([nbatch, n_capsules, n_capsules], dtype=np.float32)
            h4 = tf.reshape(h3, shape=[nbatch, n_capsules, X_size])
            h5 = routing(inputs=h4, b_IJ=b_IJ, output_size=X_size)
            assert h5.shape == [nbatch, 1, n_capsules, X_size, 1], (h5.shape, [nbatch, 1, n_capsules, X_size, 1])
            h5 = tf.reshape(h5, shape=[nbatch, n_capsules * X_size])

            pi = fc(h5, 'pi', actdim, act=lambda x: x, init_scale=0.01)
            h1 = fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            vf = fc(h5, 'vf', 1, act=lambda x: x)[:, 0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                                     initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        # v0 = vf[0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, size_mem * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, vf, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(vf, {X: ob, S: state, M: mask})

        # def step(ob, *_args, **_kwargs):
        #     a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
        #     return a, v, self.initial_state, neglogp
        #
        # def value(ob, *_args, **_kwargs):
        #     return sess.run(vf, {X: ob})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class LstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, size_mem=256, reuse=False):  # pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        if ac_space.shape == ():
            actdim = 1
        else:
            actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs

        nenv = nbatch // nsteps
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, size_mem * 2])  # states

        with tf.variable_scope("model", reuse=reuse):
            # h1 = fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            # h2 = fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)

            h2 = tf.cast(X, tf.float32)
            xs = batch_to_seq(h2, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            # h5, snew = lstm(xs, ms, S, 'lstm', nh=size_mem)
            # h5 = seq_to_batch(h5)
            h5 = h2
            h5 = tf.expand_dims(h5, axis=1)

            # state_tuple = LSTMStateTuple(*tf.split(value=S, num_or_size_splits=2, axis=1)) # input to LSTM
            # state = tf.reshape(S, shape=[nenv, size_mem, 2])
            # state = tf.transpose(state, [1, 0, 2])
            cell = LSTMCell(size_mem)
            state_tuple = cell.zero_state(batch_size=nbatch, dtype=tf.float32)
            # state_tuple = cell.zero_state(batch_size, dtype)
            h5, s_out = tf.nn.dynamic_rnn(cell, h5, dtype=tf.float32,
                                          initial_state=state_tuple)
            # s_out = tf.stack(state_tuple)  # output of LSTM

            snew = tf.reshape(tf.transpose(s_out, [1, 0, 2]), shape=[nenv, -1])
            h5 = tf.squeeze(h5, axis=1)

            pi = fc(h5, 'pi', actdim, act=lambda x: x, init_scale=0.01)
            h1 = fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            vf = fc(h5, 'vf', 1, act=lambda x: x)[:, 0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                                     initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        # v0 = vf[0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, size_mem * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, vf, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(vf, {X: ob, S: state, M: mask})

        # def step(ob, *_args, **_kwargs):
        #     a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
        #     return a, v, self.initial_state, neglogp
        #
        # def value(ob, *_args, **_kwargs):
        #     return sess.run(vf, {X: ob})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):  # pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32) / 255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x: x, init_scale=0.01)
            vf = fc(h4, 'v', 1, act=lambda x: x)[:, 0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):  # pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        if ac_space.shape == ():
            actdim = 1
        else:
            actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        with tf.variable_scope("model", reuse=reuse):
            h1 = fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            pi = fc(h2, 'pi', actdim, act=lambda x: x, init_scale=0.01)
            h1 = fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            vf = fc(h2, 'vf', 1, act=lambda x: x)[:, 0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                                     initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
