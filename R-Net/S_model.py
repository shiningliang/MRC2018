import tensorflow as tf
import logging
import time
from basic_rnn import cudnn_gru, dot_attention, ptr_net, summ


class Model(object):
    def __init__(self, args, batch, token_embeddings=None, trainable=True, opt=True):
        self.logger = logging.getLogger('brc')
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.layer_num = args.layer_num
        self.optim_type = args.optim
        self.dropout_keep_prob = args.dropout_keep_prob
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.trainable = trainable
        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len

        self.p, self.q, self.start_id, self.end_id, self.qa_id = batch.get_next()
        self.lr = tf.get_variable('lr', shape=[], dtype=tf.float32, trainable=False)
        self.is_train = tf.get_variable('is_train', shape=[], dtype=tf.bool, trainable=False)
        self.p_mask = tf.cast(self.p, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        # passage的真实长度
        self.p_len = tf.reduce_sum(tf.cast(self.p_mask, tf.int32), axis=1)
        # self.p = tf.boolean_mask(self.p, mask=self.p_len)
        # question的真实长度
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
        # self.q = tf.boolean_mask(self.q, mask=self.q_len)

        if opt:
            self.N = tf.shape(self.start_id)[0]
            # 当前batch中passage的最大长度
            self.p_maxlen = tf.reduce_max(self.p_len)
            # 当前batch中question的最大长度
            self.q_maxlen = tf.reduce_max(self.q_len)
            self.p = tf.slice(self.p, [0, 0], [self.N, self.p_maxlen])
            self.q = tf.slice(self.q, [0, 0], [self.N, self.q_maxlen])
            self.p_mask = tf.slice(self.p_mask, [0, 0], [self.N, self.p_maxlen])
            self.q_mask = tf.slice(self.q_mask, [0, 0], [self.N, self.q_maxlen])
        else:
            self.p_maxlen, self.q_maxlen = self.max_p_len, self.max_q_len

        self._build_graph(token_embeddings)

    def _build_graph(self, token_embeddings):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        # 对paragraph question做embedding
        self._embed(token_embeddings)
        # 对paragraph question分别用Bi-LSTM编码
        self._encode()
        # 基于question-aware的passage编码
        self._gated_attention()
        self._self_attention()
        self._pointer()
        # self._predict()
        # 对数似然损失，start end两部分损失取平均
        self._compute_loss()
        if self.trainable:
            # 选择优化算法
            self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))

    def _embed(self, token_embeddings):
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding', reuse=tf.AUTO_REUSE):
            word_embeddings = tf.get_variable('word_embeddings',
                                              initializer=tf.constant(token_embeddings, dtype=tf.float32),
                                              trainable=False)
            self.p_emb = tf.nn.embedding_lookup(word_embeddings, self.p)
            self.q_emb = tf.nn.embedding_lookup(word_embeddings, self.q)

    def _encode(self):
        with tf.variable_scope('passage_encoding', reuse=tf.AUTO_REUSE):
            self.p_rnn = cudnn_gru(num_layers=2*self.layer_num, num_units=self.hidden_size, batch_size=self.N,
                                   input_size=self.p_emb.get_shape().as_list()[-1],
                                   keep_prob=self.dropout_keep_prob, is_train=self.is_train)
            self.p_encodes = self.p_rnn(self.p_emb, seq_len=self.p_len)
        with tf.variable_scope('question_encoding', reuse=tf.AUTO_REUSE):
            self.q_rnn = cudnn_gru(num_layers=2*self.layer_num, num_units=self.hidden_size, batch_size=self.N,
                                   input_size=self.q_emb.get_shape().as_list()[-1],
                                   keep_prob=self.dropout_keep_prob, is_train=self.is_train)
            self.q_encodes = self.q_rnn(self.q_emb, seq_len=self.q_len)

    def _gated_attention(self):
        with tf.variable_scope('gated_attention', reuse=tf.AUTO_REUSE):
            self.qp_att = dot_attention(self.p_encodes, self.q_encodes, mask=self.q_mask,
                                        hidden=self.hidden_size, keep_prob=self.dropout_keep_prob,
                                        is_train=self.is_train)
            gated_rnn = cudnn_gru(num_layers=self.layer_num, num_units=self.hidden_size, batch_size=self.N,
                                  input_size=self.qp_att.get_shape().as_list()[-1], keep_prob=self.dropout_keep_prob,
                                  is_train=self.is_train)
            self.gated_att = gated_rnn(self.qp_att, self.p_len)  # v_Pt

    def _self_attention(self):
        with tf.variable_scope('self_attention', reuse=tf.AUTO_REUSE):
            self.pp_att = dot_attention(self.gated_att, self.gated_att, mask=self.p_mask,
                                        hidden=self.hidden_size, keep_prob=self.dropout_keep_prob,
                                        is_train=self.is_train)
            self_rnn = cudnn_gru(num_layers=self.layer_num, num_units=self.hidden_size, batch_size=self.N,
                                 input_size=self.pp_att.get_shape().as_list()[-1], keep_prob=self.dropout_keep_prob,
                                 is_train=self.is_train)
            self.self_att = self_rnn(self.pp_att, self.p_len)

    def _pointer(self):
        with tf.variable_scope('pointer', reuse=tf.AUTO_REUSE):
            self.ques_vec = summ(self.q_encodes[:, :, -2 * self.hidden_size:], self.hidden_size, mask=self.q_mask,
                                 keep_prob=self.dropout_keep_prob, is_train=self.is_train)  # r_Q
            pointer = ptr_net(batch=self.N, hidden=self.ques_vec.get_shape().as_list()[-1],
                              keep_prob=self.dropout_keep_prob, is_train=self.is_train)
            self.logits1, self.logits2 = pointer(self.ques_vec, self.self_att, self.hidden_size, self.p_mask)

    def _predict(self):
        with tf.variable_scope("predict"):
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(self.logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(self.logits2), axis=1))
            self.outer = tf.matrix_band_part(outer, 0, self.max_a_len)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            self.start_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits1, labels=tf.stop_gradient(
                tf.one_hot(self.start_id, tf.shape(self.logits1)[1], axis=1)))
            self.end_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits2, labels=tf.stop_gradient(
                tf.one_hot(self.start_id, tf.shape(self.logits2)[1], axis=1)))
            self.loss = tf.reduce_mean(self.start_loss + self.end_loss)

    def _compute_loss(self):
        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses
        self.logits1 = tf.nn.softmax(self.logits1)
        self.logits2 = tf.nn.softmax(self.logits2)
        self.start_loss = sparse_nll_loss(probs=self.logits1, labels=self.start_id)
        self.end_loss = sparse_nll_loss(probs=self.logits2, labels=self.end_id)
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss

    def _create_train_op(self):
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            if self.optim_type == 'adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(self.lr)
            elif self.optim_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.optim_type == 'rprop':
                self.optimizer = tf.train.RMSPropOptimizer(self.lr)
            elif self.optim_type == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
            else:
                raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
            self.train_op = self.optimizer.minimize(self.loss)
