"""
This module provides wrappers for variants of RNN in Tensorflow
"""

import tensorflow as tf
from tensorflow.contrib import cudnn_rnn


def rnn(rnn_type, inputs, hidden_size, batch_size, training, layer_num=1, dropout_keep_prob=None):
    """
    Implements (Bi-)LSTM, (Bi-)GRU and (Bi-)RNN
    Args:
        rnn_type: the type of rnn
        inputs: padded inputs into rnn
        hidden_size: the size of hidden units
        layer_num: multiple rnn layer are stacked if layer_num > 1
        dropout_keep_prob:
    Returns:
        RNN outputs and final state
    """
    if not rnn_type.startswith('bi'):
        cell = get_cell(rnn_type, hidden_size, layer_num, 'unidirectional')
        inputs = tf.transpose(inputs, [1, 0, 2])
        c = tf.zeros([layer_num, batch_size, hidden_size], tf.float32)
        h = tf.zeros([layer_num, batch_size, hidden_size], tf.float32)
        outputs, state = cell(inputs, (h, c), training=training)
        if rnn_type.endswith('lstm'):
            c, h = state
            state = h
    else:
        cell = get_cell(rnn_type, hidden_size, layer_num, 'bidirectional')
        inputs = tf.transpose(inputs, [1, 0, 2])
        outputs, state = cell(inputs, training=training)
        # if rnn_type.endswith('lstm'):
        #     state_h, state_c = state
        #     h_fw, h_bw = state_h[0, :], state_h[1, :]
        #     state_fw, state_bw = h_fw, h_bw
        # else:
        #     state_fw, state_bw = state[0][0, :], state[0][1, :]
        # if concat:
        #     state = tf.concat([state_fw, state_bw], 1)
        # else:
        #     state = state_fw + state_bw
    outputs = tf.transpose(outputs, [1, 0, 2])
    return outputs, state


def get_cell(rnn_type, hidden_size, layer_num=1, direction='bidirectional'):
    if rnn_type.endswith('lstm'):
        cudnn_cell = cudnn_rnn.CudnnLSTM(num_layers=layer_num, num_units=hidden_size, direction=direction,
                                         dropout=0)
    elif rnn_type.endswith('gru'):
        cudnn_cell = cudnn_rnn.CudnnGRU(num_layers=layer_num, num_units=hidden_size, direction=direction,
                                        dropout=0)
    elif rnn_type.endswith('rnn'):
        cudnn_cell = cudnn_rnn.CudnnRNNTanh(num_layers=layer_num, num_units=hidden_size, direction=direction,
                                            dropout=0)
    else:
        raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
    return cudnn_cell
