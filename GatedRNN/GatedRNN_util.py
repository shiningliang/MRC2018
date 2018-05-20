import tensorflow as tf
import numpy as np
import os
import ujson as json
from sklearn.metrics import accuracy_score
TYPE = {0: 'Yes', 1: 'No', 2: 'Depends'}


def get_record_parser(config):
    def parse(example):
        ans_limit = config.max_a_len
        ques_limit = config.max_q_len
        features = tf.parse_single_example(example,
                                           features={
                                               'answer_token_ids': tf.FixedLenFeature([], tf.string),
                                               'question_token_ids': tf.FixedLenFeature([], tf.string),
                                               'answer_type': tf.FixedLenFeature([], tf.string),
                                               'id': tf.FixedLenFeature([], tf.int64)
                                           })
        answer_token_ids = tf.reshape(tf.decode_raw(features['answer_token_ids'], tf.int32), [ans_limit])
        question_token_ids = tf.reshape(tf.decode_raw(features['question_token_ids'], tf.int32), [ques_limit])
        answer_type = tf.reshape(tf.decode_raw(features['answer_type'], tf.int32), [3])
        qa_id = features['id']
        return answer_token_ids, question_token_ids, answer_type, qa_id

    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(parser, num_parallel_calls=num_threads).shuffle(
        config.capacity).batch(config.batch_size).repeat(config.epochs)
    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).batch(config.batch_size).repeat()
    return dataset


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    losses = []
    # pred_answers = []
    pre_ans_types, ref_ans_types = [], []
    for i in range(num_batches):
        qa_ids, loss, pre_labels = sess.run([model.qa_id, model.loss, model.pre_labels],
                                            feed_dict={handle: str_handle} if handle is not None else None)
        losses.append(loss)
        for qa_id, pre_label in zip(qa_ids, pre_labels):
            sample = eval_file[str(qa_id)]
            pre_ans_types.append(pre_label)
            ref_ans_types.append(sample['answer_type'])

    avg_loss = np.mean(losses)
    avg_acc = accuracy_score(y_true=ref_ans_types, y_pred=pre_ans_types)

    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/loss".format(data_type), simple_value=avg_loss), ])
    acc_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/f1".format(data_type), simple_value=avg_acc), ])
    return avg_loss, avg_acc, [loss_sum, acc_sum]


def predict_batch(model, num_batches, eval_file, sess, data_type, final_file, logger):
    pred_answers = []
    for i in range(num_batches):
        qa_ids, pre_labels = sess.run([model.qa_id, model.pre_labels])
        for qa_id, pre_label in zip(qa_ids, pre_labels):
            sample = eval_file[str(qa_id)]
            pred_answers.append({'question_id': sample['question_id'],
                                 'question_type': 'YES_NO',
                                 'answers': sample['answers'],
                                 'entity_answers': [[]],
                                 'yesno_answers': [TYPE[pre_label]]})

    logger.info('{} questions'.format(len(pred_answers)))
    with open(final_file, 'a') as fout:
        for pred_answer in pred_answers:
            fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')
    fout.close()
    logger.info('Saving classification results')
