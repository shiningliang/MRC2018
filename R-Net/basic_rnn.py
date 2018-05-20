import tensorflow as tf
import tensorflow.contrib as tc

INF = 1e30


class cudnn_gru:
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tc.cudnn_rnn.CudnnGRU(1, num_units)
            gru_bw = tc.cudnn_rnn.CudnnGRU(1, num_units)

            init_fw = tf.tile(tf.Variable(tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            init_bw = tf.tile(tf.Variable(tf.zeros([1, 1, num_units])), [1, batch_size, 1])

            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw,))
            self.inits.append((init_fw, init_bw,))
            self.dropout_mask.append((mask_fw, mask_bw,))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=False):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            init_fw, init_bw = self.inits[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]

            with tf.variable_scope("fw_{}".format(layer)):
                out_fw, _ = gru_fw(outputs[-1] * mask_fw, initial_state=(init_fw,))

            with tf.variable_scope("bw_{}".format(layer)):
                # 将输入逆序，作为backward的输入
                inputs_bw = tf.reverse_sequence(outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                out_bw, _ = gru_bw(inputs_bw, initial_state=(init_bw,))
                # 输出逆序，与forward concat后作为下一层的输入
                out_bw = tf.reverse_sequence(out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))

        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res


def rnn(rnn_type, inputs, length, hidden_size, layer_num=1, dropout_keep_prob=None, concat=True):
    if not rnn_type.startswith('bi'):
        cell = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=length, dtype=tf.float32)
        if rnn_type.endswith('lstm'):
            c, h = state
            state = h
    else:
        cell_fw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
        cell_bw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
        outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_bw, cell_fw, inputs,
                                                         sequence_length=length, dtype=tf.float32)
        state_fw, state_bw = state
        if rnn_type.endswith('lstm'):
            c_fw, h_fw = state_fw
            c_bw, h_bw = state_bw
            state_fw, state_bw = h_fw, h_bw
        if concat:
            outputs = tf.concat(outputs, 2)
            state = tf.concat([state_fw, state_bw], 1)
        else:
            outputs = outputs[0] + outputs[1]
            state = state_fw + state_bw
    return outputs, state


def get_cell(rnn_type, hidden_size, layer_num=1, dropout_keep_prob=None):
    if rnn_type.endswith('lstm'):
        cell = tc.rnn.LSTMBlockCell(num_units=hidden_size, reuse=tf.AUTO_REUSE)
        # cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    elif rnn_type.endswith('gru'):
        cell = tc.rnn.GRUBlockCell(num_units=hidden_size, reuse=tf.AUTO_REUSE)
        # cell = tc.rnn.GRUCell(num_units=hidden_size)
    elif rnn_type.endswith('rnn'):
        cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
    else:
        raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
    # if dropout_keep_prob is not None:
    #     cell = tc.rnn.DropoutWrapper(cell,
    #                                  input_keep_prob=dropout_keep_prob,
    #                                  output_keep_prob=dropout_keep_prob)
    if layer_num > 1:
        cell = tc.rnn.MultiRNNCell([cell] * layer_num, state_is_tuple=True)
    return cell


def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None):
    d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)  # u_pt
    d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)  # u_qt
    JX = tf.shape(inputs)[1]  # passage_len

    with tf.variable_scope("attention"):
        inputs_ = tf.nn.relu(dense(d_inputs, hidden, use_bias=False, scope="inputs"))
        memory_ = tf.nn.relu(dense(d_memory, hidden, use_bias=False, scope="memory"))
        outputs = tf.matmul(inputs_, tf.transpose(memory_, [0, 2, 1])) / (hidden ** 0.5)  # s_tj
        # mask重复passage_len次，即passage中每个word相对于整个question mask=N*passage_len*question_len
        mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
        logits = tf.nn.softmax(softmax_mask(outputs, mask))
        outputs = tf.matmul(logits, memory)  # c_t
        res = tf.concat([inputs, outputs], axis=2)  # [u_Pt, c_t]

    with tf.variable_scope("gate"):
        dim = res.get_shape().as_list()[-1]
        d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
        gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))
        return res * gate


def softmax_mask(val, mask):
    return -INF * (1 - tf.cast(mask, tf.float32)) + val


def dropout(args, keep_prob, is_train, mode="recurrent"):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        args = tf.cond(is_train, lambda: tf.nn.dropout(args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args


def dense(inputs, hidden, use_bias=True, scope="dense"):
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        # 前两个维度与输入相同，最后加上输出维度
        out_shape = [shape[idx] for idx in range(len(inputs.get_shape().as_list()) - 1)] + [hidden]

        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable("b", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res


class ptr_net:
    def __init__(self, batch, hidden, keep_prob=1.0, is_train=None, scope="ptr_net"):
        self.gru = tf.contrib.rnn.GRUCell(hidden)
        self.batch = batch
        self.scope = scope
        self.keep_prob = keep_prob
        self.is_train = is_train
        self.dropout_mask = dropout(tf.ones([batch, hidden], dtype=tf.float32),
                                    keep_prob=keep_prob, is_train=is_train)

    def __call__(self, init, match, d, mask):
        # match = h_P
        with tf.variable_scope(self.scope):
            # t=1
            d_match = dropout(match, keep_prob=self.keep_prob,
                              is_train=self.is_train)  # h_P
            inp, logits1 = pointer(d_match, init * self.dropout_mask, d, mask)
            d_inp = dropout(inp, keep_prob=self.keep_prob,
                            is_train=self.is_train)  # c_1
            _, state = self.gru(d_inp, init)  # h_a1
            tf.get_variable_scope().reuse_variables()
            # t=2
            _, logits2 = pointer(d_match, state * self.dropout_mask, d, mask)
            return logits1, logits2


def pointer(inputs, state, hidden, mask, scope="pointer"):
    """
    pointer net core
    :param inputs: h_Pj
    :param state: h_a(t-1)
    :param hidden:
    :param mask:
    :param scope:
    :return: c_t p_t
    """
    with tf.variable_scope(scope):
        u = tf.concat([tf.tile(tf.expand_dims(state, axis=1), [1, tf.shape(inputs)[1], 1]), inputs], axis=2)
        s0 = tf.nn.tanh(dense(u, hidden, use_bias=False, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)  # s_tj
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * inputs, axis=1)
        return res, s1


def summ(memory, hidden, mask, keep_prob=1.0, is_train=None, scope="summ"):
    with tf.variable_scope(scope):
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)  # u_Qj
        s0 = tf.nn.tanh(dense(d_memory, hidden, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * memory, axis=1)
        return res
