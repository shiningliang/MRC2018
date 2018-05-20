import tensorflow as tf
import logging
import time
import os
from basic_rnn import dot_attention, dense, cudnn_gru


class GatedRNN(object):
    def __init__(self, args, batch, token_embeddings=None, trainable=True, opt=True):
        # logging
        self.logger = logging.getLogger("brc")
        # basic config
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.output_size = 3
        # self.d_a = 300
        # self.r = 64
        # self.p_coef = 1
        self.layer_num = args.layer_num
        self.optim_type = args.optim
        self.weight_decay = args.weight_decay
        self.dropout_keep_prob = args.dropout_keep_prob
        self.trainable = trainable
        # length limit
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len
        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self.a, self.q, self.answers_type, self.qa_id = batch.get_next()
        self.lr = tf.get_variable('lr', shape=[], dtype=tf.float32, trainable=False)
        self.is_train = tf.get_variable('is_train', shape=[], dtype=tf.bool, trainable=False)
        self.a_mask = tf.cast(self.a, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.a_len = tf.reduce_sum(tf.cast(self.a_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
        self.N = tf.shape(self.qa_id)[0]

        self._build_graph(token_embeddings)

    def _build_graph(self, token_embeddings):
        start_t = time.time()
        self._embed(token_embeddings)
        self._encode()
        self._gated_attention()
        self._self_attention()
        # self._annotation()
        self._predict()
        self._compute_loss()
        if self.trainable:
            self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))

    def _embed(self, token_embeddings):
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding', reuse=tf.AUTO_REUSE):
            word_embeddings = tf.get_variable('word_embeddings',
                                              initializer=tf.constant(token_embeddings, dtype=tf.float32),
                                              trainable=False)
            self.a_emb = tf.nn.embedding_lookup(word_embeddings, self.a)
            self.q_emb = tf.nn.embedding_lookup(word_embeddings, self.q)

    def _encode(self):
        with tf.variable_scope('answer_encoding', reuse=tf.AUTO_REUSE):
            a_rnn = cudnn_gru(num_layers=2 * self.layer_num, num_units=self.hidden_size, batch_size=self.N,
                              input_size=self.a_emb.get_shape().as_list()[-1],
                              keep_prob=self.dropout_keep_prob, is_train=self.is_train)
            self.a_encodes = a_rnn(self.a_emb, seq_len=self.a_len)
        with tf.variable_scope('question_encoding', reuse=tf.AUTO_REUSE):
            q_rnn = cudnn_gru(num_layers=2 * self.layer_num, num_units=self.hidden_size, batch_size=self.N,
                              input_size=self.q_emb.get_shape().as_list()[-1],
                              keep_prob=self.dropout_keep_prob, is_train=self.is_train)
            self.q_encodes = q_rnn(self.q_emb, seq_len=self.q_len)

    def _gated_attention(self):
        with tf.variable_scope('gated_attention', reuse=tf.AUTO_REUSE):
            self.qa_att = dot_attention(self.a_encodes, self.q_encodes, mask=self.q_mask,
                                        hidden=self.hidden_size, keep_prob=self.dropout_keep_prob,
                                        is_train=self.is_train)
            gated_rnn = cudnn_gru(num_layers=self.layer_num, num_units=self.hidden_size, batch_size=self.N,
                                  input_size=self.qa_att.get_shape().as_list()[-1], keep_prob=self.dropout_keep_prob,
                                  is_train=self.is_train)
            self.gated_att = gated_rnn(self.qa_att, self.a_len)

    def _self_attention(self):
        with tf.variable_scope('self_attention', reuse=tf.AUTO_REUSE):
            self.aa_att = dot_attention(self.gated_att, self.gated_att, mask=self.a_mask,
                                        hidden=self.hidden_size, keep_prob=self.dropout_keep_prob,
                                        is_train=self.is_train)
            self_rnn = cudnn_gru(num_layers=self.layer_num, num_units=self.hidden_size, batch_size=self.N,
                                 input_size=self.aa_att.get_shape().as_list()[-1], keep_prob=self.dropout_keep_prob,
                                 is_train=self.is_train)
            self.self_att = self_rnn(self.aa_att, self.a_len)

    def _annotation(self):
        # shape(W_s1) = d_a * 2u
        self.W_s1 = tf.get_variable('W_s1', shape=[self.d_a, 2 * self.hidden_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
        # shape(W_s2) = r * d_a
        self.W_s2 = tf.get_variable('W_s2', shape=[self.r, self.d_a],
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.A = tf.nn.softmax(tf.map_fn(
            lambda x: tf.matmul(self.W_s2, x),
            tf.tanh(tf.map_fn(lambda x: tf.matmul(self.W_s1, tf.transpose(x)),
                              self.gated_att))))
        self.M = tf.matmul(self.A, self.gated_att)
        self.A_T = tf.transpose(self.A, perm=[0, 2, 1])
        tile_eye = tf.tile(tf.eye(self.r), [self.N, 1])
        tile_eye = tf.reshape(tile_eye, [-1, self.r, self.r])
        self.AA_T = tf.matmul(self.A, self.A_T) - tile_eye
        self.P = tf.square(tf.norm(self.AA_T, axis=[-2, -1], ord='fro'))

    def _predict(self):
        with tf.variable_scope('predict', reuse=tf.AUTO_REUSE):
            self.att = tf.reshape(self.self_att, shape=[self.N, 2 * self.max_a_len * self.hidden_size])
            self.mlp = tf.nn.relu(dense(self.att, hidden=4 * self.hidden_size, scope='dense_0'))
            if self.is_train:
                self.mlp = tf.nn.dropout(self.mlp, self.dropout_keep_prob)
            self.mlp = tf.nn.relu(dense(self.mlp, hidden=2 * self.hidden_size, scope='dense_1'))
            if self.is_train:
                self.mlp = tf.nn.dropout(self.mlp, self.dropout_keep_prob)
            self.outputs = dense(self.mlp, hidden=self.output_size, scope='output')

    def _compute_loss(self):
        self.pre_labels = tf.argmax(self.outputs, axis=1)
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.outputs,
                                                                                            labels=tf.stop_gradient(
                                                                                                self.answers_type))))

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
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
