"""
This module provides wrappers for variants of RNN in Tensorflow
"""

import tensorflow as tf
import tensorflow.contrib as tc


def rnn(rnn_type, inputs, length, hidden_size, layer_num=1, dropout_keep_prob=None, concat=True):
    """
    Implements (Bi-)LSTM, (Bi-)GRU and (Bi-)RNN
    Args:
        rnn_type: the type of rnn
        inputs: padded inputs into rnn
        length: the valid length of the inputs
        hidden_size: the size of hidden units
        layer_num: multiple rnn layer are stacked if layer_num > 1
        dropout_keep_prob:
        concat: When the rnn is bidirectional, the forward outputs and backward outputs are
                concatenated if this is True, else we add them.
    Returns:
        RNN outputs and final state
    """
    if not rnn_type.startswith('bi'):
        cells = tc.rnn.MultiRNNCell([get_nor_cell(rnn_type, hidden_size, dropout_keep_prob) for _ in range(layer_num)],
                                    state_is_tuple=True)
        outputs, state = tf.nn.dynamic_rnn(cells, inputs, sequence_length=length, dtype=tf.float32)
        if rnn_type.endswith('lstm'):
            c, h = state
            state = h
    else:
        if layer_num > 1:
            cell_fw = [get_nor_cell(rnn_type, hidden_size, dropout_keep_prob) for _ in range(layer_num)]
            cell_bw = [get_nor_cell(rnn_type, hidden_size, dropout_keep_prob) for _ in range(layer_num)]
            outputs, state_fw, state_bw = tc.rnn.stack_bidirectional_dynamic_rnn(
                cell_fw, cell_bw, inputs, sequence_length=length, dtype=tf.float32
            )
        else:
            cell_fw = get_nor_cell(rnn_type, hidden_size, dropout_keep_prob)
            cell_bw = get_nor_cell(rnn_type, hidden_size, dropout_keep_prob)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, inputs, sequence_length=length, dtype=tf.float32
            )
    return outputs


def get_cell(rnn_type, hidden_size, layer_num=1, dropout_keep_prob=None):
    """
    Gets the RNN Cell
    Args:
        rnn_type: 'lstm', 'gru' or 'rnn'
        hidden_size: The size of hidden units
        layer_num: MultiRNNCell are used if layer_num > 1
        dropout_keep_prob: dropout in RNN
    Returns:
        An RNN Cell
    """
    if rnn_type.endswith('lstm'):
        cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    elif rnn_type.endswith('gru'):
        cell = tc.rnn.GRUCell(num_units=hidden_size)
    elif rnn_type.endswith('rnn'):
        cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
    elif rnn_type.endswith('sru'):
        cell = tc.rnn.SRUCell(num_units=hidden_size)
    elif rnn_type.endswith('indy'):
        cell = tc.rnn.IndyGRUCell(num_units=hidden_size)
    else:
        raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
    if dropout_keep_prob is not None:
        cell = tc.rnn.DropoutWrapper(cell,
                                     input_keep_prob=dropout_keep_prob,
                                     output_keep_prob=dropout_keep_prob)
    return cell


