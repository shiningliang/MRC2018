# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
"""

import os
import time
import logging
import ujson as json
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from utils import mrc_eval
from utils.bleu import BLEUWithBonus
from utils.rouge import RougeLWithBonus
from layers.cu_rnn import rnn
from layers.match_layer import MatchLSTMLayer, AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder


class RCModel(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, embeddings, pad_id, args, train=True):

        # logging
        self.logger = logging.getLogger("brc")

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.layer_num = args.layer_num
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1
        self.is_training = train
        self.model_dir = args.model_dir
        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len
        # the vocab
        # self.vocab = vocab
        self.embeddings = embeddings
        self.pad_id = pad_id
        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        # if train:
        self._build_graph()
        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        # 设置需传入的常量
        self._setup_placeholders()
        # 对paragraph question做embedding
        self._embed()
        # 对paragraph question分别用Bi-LSTM编码
        self._encode()
        # 基于question-aware的passage编码
        self._match()
        # 融合上下文的match passage再编码
        self._fuse()
        # 使用pointer network计算每个位置为答案开头或结尾的概率
        self._decode()
        # 对数似然损失，start end两部分损失取平均
        self._compute_loss()
        if self.is_training:
            # 选择优化算法
            self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        # param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        # self.logger.info('There are {} parameters in the model'.format(param_num))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self._lr = tf.Variable(0.0, trainable=False)
        self._new_lr = tf.placeholder(tf.float32)
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding', reuse=tf.AUTO_REUSE):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                # shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant(self.embeddings, dtype=tf.float32),
                trainable=False,
            )
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)

    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        with tf.variable_scope('passage_encoding', reuse=tf.AUTO_REUSE):
            self.sep_p_encodes, _ = rnn('bi-lstm', self.p_emb, self.hidden_size, self.batch_size, self.is_training,
                                        layer_num=self.layer_num, dropout_keep_prob=self.dropout_keep_prob)
            # self.sep_p_encodes, _ = rnn('bi-lstm', self.p_emb, self.p_length, self.hidden_size, self.layer_num)
        with tf.variable_scope('question_encoding', reuse=tf.AUTO_REUSE):
            self.sep_q_encodes, _ = rnn('bi-lstm', self.q_emb, self.hidden_size, self.batch_size, self.is_training,
                                        layer_num=self.layer_num, dropout_keep_prob=self.dropout_keep_prob)
            # self.sep_q_encodes, _ = rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size, self.layer_num)
        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)

    def _match(self):
        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """
        if self.algo == 'MLSTM':
            self.match_layer = MatchLSTMLayer(self.hidden_size)
        elif self.algo == 'BIDAF':
            self.match_layer = AttentionFlowMatchLayer(self.hidden_size)
        else:
            raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))
        self.match_p_encodes, _ = self.match_layer.match(self.sep_p_encodes, self.sep_q_encodes,
                                                         self.p_length, self.q_length)
        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        with tf.variable_scope('fusion', reuse=tf.AUTO_REUSE):
            self.fuse_p_encodes, _ = rnn('bi-lstm', self.match_p_encodes, self.hidden_size, self.batch_size,
                                         self.is_training, layer_num=self.layer_num,
                                         dropout_keep_prob=self.dropout_keep_prob)
            # self.fuse_p_encodes, _ = rnn('bi-lstm', self.match_p_encodes, self.p_length, self.hidden_size)
            if self.use_dropout:
                self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, self.dropout_keep_prob)

    def _decode(self):
        """
        Employs Pointer Network to get the the probs of each position
        to be the start or end of the predicted answer.
        Note that we concat the fuse_p_encodes for the passages in the same document.
        And since the encodes of queries in the same document is same, we select the first one.
        """
        with tf.variable_scope('same_question_concat', reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(self.start_label)[0]
            concat_passage_encodes = tf.reshape(
                self.fuse_p_encodes,
                [batch_size, -1, 2 * self.hidden_size]
            )
            no_dup_question_encodes = tf.reshape(
                self.sep_q_encodes,
                [batch_size, -1, tf.shape(self.sep_q_encodes)[1], 2 * self.hidden_size]
            )[0:, 0, 0:, 0:]
        self.decoder = PointerNetDecoder(self.hidden_size)
        self.start_probs, self.end_probs = self.decoder.decode(concat_passage_encodes,
                                                               no_dup_question_encodes)

    def _compute_loss(self):
        def sparse_nll_loss(probs, labels, epsilon=1e-9, gamma=2.0, alpha=2.0, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                # model_out = tf.add(probs, epsilon)
                # labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                # ce = tf.multiply(labels, -tf.log(model_out))
                # weight = tf.multiply(labels, tf.pow(tf.subtract(1., model_out), gamma))
                # fl = tf.multiply(alpha, tf.multiply(weight, ce))
                # reduced_fl = tf.reduce_sum(fl, axis=1)
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses

        self.start_loss = sparse_nll_loss(probs=self.start_probs, labels=self.start_label)
        self.end_loss = sparse_nll_loss(probs=self.end_probs, labels=self.end_label)
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            if self.optim_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self._lr)
            elif self.optim_type == 'adam':
                self.optimizer = tc.opt.LazyAdamOptimizer(self.learning_rate)
            elif self.optim_type == 'rprop':
                self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif self.optim_type == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            else:
                raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
            self.train_op = self.optimizer.minimize(self.loss)

    def _train_epoch(self, train_batches, dropout_keep_prob, epoch):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        for bitx, batch in enumerate(train_batches, 1):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: dropout_keep_prob}
            lr_decay = 0.95 ** max(epoch - 5, 0)
            self._assign_lr(self.sess, self.learning_rate * lr_decay)
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
            # if bitx == 4000:
            #     return 1.0 * total_loss / total_num
        return 1.0 * total_loss / total_num

    def _assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def train(self, data, epochs, batch_size, save_dir, save_prefix, dropout_keep_prob=1.0, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        max_Rouge_L = 0
        # saver = tf.train.Saver()
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train', batch_size, self.pad_id, shuffle=True)
            # 这里传入当前epoch数
            train_loss = self._train_epoch(train_batches, dropout_keep_prob, epoch)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))
            del train_batches

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, self.pad_id, shuffle=False)
                    # 使用验证集评价模型
                    eval_loss, bleu_rouge = self.evaluate(eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))
                    del eval_batches

                    if bleu_rouge['Rouge-L'] > max_Rouge_L:
                        test_batches = data.gen_mini_batches('test', batch_size, self.pad_id, shuffle=False)
                        self.evaluate(test_batches, result_dir=save_dir, result_prefix=save_prefix)
                        max_Rouge_L = bleu_rouge['Rouge-L']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: 1.0}
            # 得到输出
            start_probs, end_probs, loss = self.sess.run([self.start_probs,
                                                          self.end_probs, self.loss], feed_dict)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            padded_p_len = len(batch['passage_token_ids'][0])
            for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):

                best_answer = self.find_best_answer(sample, start_prob, end_prob, padded_p_len)
                # 如果保存全部信息，则在样本中加入pred_answers，否则只保留部分信息，答案放入answers
                # 预测答案
                if save_full_info:
                    sample['pred_answers'] = [best_answer]
                    pred_answers.append(sample)
                else:
                    pred_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': [best_answer],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})
                # 标准答案
                if 'answers' in sample:
                    ref_answers.append({'question_id': sample['question_id'],
                                        'question_type': sample['question_type'],
                                        'answers': sample['answers'],
                                        'entity_answers': [[]],
                                        'yesno_answers': []})

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            # K-V 问题ID-答案
            pred_dict, ref_dict, bleu_rouge = {}, {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    # 将answer tokens转换为由空格连接的一句话
                    pred_dict[question_id] = {'answers': mrc_eval.normalize(pred['answers']),
                                              'entity_answers': [[]],
                                              'yesno_answers': []}
                    ref_dict[question_id] = {'question_type': ref['question_type'],
                                             'answers': mrc_eval.normalize(ref['answers']),
                                             'entity_answers': [[]],
                                             'yesno_answers': []}
            bleu_eval = BLEUWithBonus(4, alpha=1.0, beta=1.0)
            rouge_eval = RougeLWithBonus(alpha=1.0, beta=1.0, gamma=1.2)
            bleu4, rouge_l = mrc_eval.calc_metrics(pred_dict,
                                                   ref_dict,
                                                   bleu_eval,
                                                   rouge_eval)
            bleu_rouge['Bleu-4'] = bleu4
            bleu_rouge['Rouge-L'] = rouge_l
            # bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        return ave_loss, bleu_rouge

    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len):
        """
        Finds the best answer for a sample given start_prob and end_prob for each position.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['passages']):
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            # 为每个passage找到best answer
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            # 各passage间最大score
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        # 根据span找到token
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            best_answer = ''.join(
                sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        return best_answer

    def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob from a single passage
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        # 从头扫描passage
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        file_path = os.path.join(model_dir, model_prefix)
        # self.saver = tf.train.import_meta_graph(file_path + '.meta')
        self.saver.restore(self.sess, file_path)
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))
