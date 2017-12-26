#! /usr/bin/env python

import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

batch_size = 8
truncated_backprop_length = 1
state_size = 16
cell = tf.nn.rnn_cell.LSTMCell(state_size)
s = tf.reshape(tf.range(batch_size * state_size * 2, dtype=tf.float32),
               shape=[batch_size, 2 * state_size])
state = LSTMStateTuple(*tf.split(s, num_or_size_splits=2, axis=1))
in_val = tf.zeros((batch_size, truncated_backprop_length, state_size))
out_val, s = tf.nn.dynamic_rnn(cell, in_val, dtype=tf.float32,
                               initial_state=state)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


x = tf.constant([
    [[1, 2, 3],
     [4, 5, 6]],
    [[11, 12, 13],
     [14, 15, 16]]
])

y = tf.transpose(x, [1, 0, 2])
y = tf.reshape(y, shape=[2, 6])

# we want to get this to
# [[1, 2, 3, 11, 12, 13],
#  [4, 5, 6, 14, 15, 16]]


print(sess.run(y))
# print(sess.run(list(map(tf.shape, [out_val, s]))))
