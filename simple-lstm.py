#! /usr/bin/env python

import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

batch_size = 8
truncated_backprop_length = 1
state_size = 16
cell = tf.nn.rnn_cell.LSTMCell(state_size)
s = tf.reshape(tf.range(batch_size * state_size * 2, dtype=tf.float32), shape=[2, batch_size, state_size])
state = LSTMStateTuple(*[
    tf.squeeze(x, axis=0) for x in tf.split(s, num_or_size_splits=2, axis=0)
])
in_val = tf.zeros((batch_size, truncated_backprop_length, state_size))
out_val, s = tf.nn.dynamic_rnn(cell, in_val, dtype=tf.float32,
                               initial_state=state)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(list(map(tf.shape, [out_val, s]))))
