import tensorflow as tf
import numpy as np
import os
import ujson as json
import mrc_eval
from bleu import BLEUWithBonus
from rouge import RougeLWithBonus


def get_record_parser(config):
    def parse(example):
        features = tf.parse_single_example(example,
                                           features={
                                               'passages_token_ids': tf.FixedLenFeature([], tf.string),
                                               'question_token_ids': tf.FixedLenFeature([], tf.string),
                                               'start_id': tf.FixedLenFeature([], tf.int64),
                                               'end_id': tf.FixedLenFeature([], tf.int64),
                                               'id': tf.FixedLenFeature([], tf.int64)
                                           })
        passages_token_ids = tf.reshape(tf.decode_raw(features["passages_token_ids"], tf.int32),
                                        [config.max_p_num * config.max_p_len])
        question_token_ids = tf.reshape(tf.decode_raw(features["question_token_ids"], tf.int32),
                                        [config.max_q_len])
        start_id = features['start_id']
        end_id = features['end_id']
        qa_id = features['id']
        return passages_token_ids, question_token_ids, start_id, end_id, qa_id

    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(parser, num_parallel_calls=num_threads).shuffle(
        config.capacity).batch(config.batch_size).repeat(config.epochs)
    # if config.is_bucket:
    #     buckets = [tf.constant(num) for num in range(*config.bucket_range)]
    #
    #     def key_func(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id):
    #         c_len = tf.reduce_sum(
    #             tf.cast(tf.cast(context_idxs, tf.bool), tf.int32))
    #         buckets_min = [np.iinfo(np.int32).min] + buckets
    #         buckets_max = buckets + [np.iinfo(np.int32).max]
    #         conditions_c = tf.logical_and(
    #             tf.less(buckets_min, c_len), tf.less_equal(c_len, buckets_max))
    #         bucket_id = tf.reduce_min(tf.where(conditions_c))
    #         return bucket_id
    #
    #     def reduce_func(key, elements):
    #         return elements.batch(config.batch_size)
    #
    #     dataset = dataset.apply(
    #         tf.contrib.data.group_by_window(key_func, reduce_func, window_size=5 * config.batch_size)).shuffle(
    #         len(buckets) * 25)
    # else:
    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).batch(config.batch_size).repeat()
    return dataset


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle, args, logger, result_prefix=None):
    losses = []
    pred_answers, ref_answers = [], []
    padded_p_len = args.max_p_len
    for i in range(num_batches):
        qa_id, loss, start_probs, end_probs = sess.run([model.qa_id, model.loss, model.logits1, model.logits2],
                                                       feed_dict={handle: str_handle} if handle is not None else None)
        losses.append(loss)
        start, end = 0, 0
        for id, start_prob, end_prob in zip(qa_id, start_probs, end_probs):
            best_p_idx, best_span, best_score = None, None, 0
            sample = eval_file[str(id)]
            for p_idx, passage_len in enumerate(sample['passages_len']):
                if p_idx >= args.max_p_num:
                    continue
                # 为每个passage找到best answer
                end = start + passage_len
                answer_span, score = find_best_answer_for_passage(start_prob[start: end], end_prob[start: end],
                                                                  passage_len, args.max_a_len)
                answer_span[0] += start
                answer_span[1] += start
                # 各passage间最大score
                if score > best_score:
                    best_score = score
                    best_p_idx = p_idx
                    best_span = answer_span
                end = start
            # best_span = [start_prob, end_prob]
            # best_answer = sample['passages'][best_span[0]: best_span[1] + 1]
            # 根据span找到token
            if best_p_idx is None or best_span is None:
                best_answer = ''
            else:
                best_answer = ''.join(sample['passages'][best_span[0]: best_span[1] + 1])
            # TODO 加入question tokens
            pred_answers.append({'question_id': sample['question_id'],
                                 'question_type': sample['question_type'],
                                 'answers': [best_answer],
                                 'yesno_answers': []})
            # 标准答案
            # if 'answers' in sample and len(sample['answers']) > 0:
            if 'answers' in sample:
                ref_answers.append({'question_id': sample['question_id'],
                                    'question_type': sample['question_type'],
                                    'answers': sample['answers'],
                                    'yesno_answers': []})

    if result_prefix is not None:
        result_file = os.path.join(args.result_dir, result_prefix + '.json')
        with open(result_file, 'w') as fout:
            for pred_answer in pred_answers:
                fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')
        logger.info('Saving {} results to {}'.format(result_prefix, result_file))

    avg_loss = np.mean(losses)
    bleu4, rouge_l = 0, 0
    if len(ref_answers) > 0:
        # K-V 问题ID-答案
        pred_dict, ref_dict, bleu_rouge = {}, {}, {}
        for pred, ref in zip(pred_answers, ref_answers):
            question_id = ref['question_id']
            if len(ref['answers']) > 0:
                # 将answer tokens转换为由空格连接的一句话
                pred_dict[question_id] = {'answers': mrc_eval.normalize(pred['answers']),
                                          'yesno_answers': []}
                ref_dict[question_id] = {'question_type': ref['question_type'],
                                         'answers': mrc_eval.normalize(ref['answers']),
                                         'yesno_answers': []}
        bleu_eval = BLEUWithBonus(4, alpha=1.0, beta=1.0)
        rouge_eval = RougeLWithBonus(alpha=1.0, beta=1.0, gamma=1.2)
        bleu4, rouge_l = mrc_eval.calc_metrics(pred_dict,
                                               ref_dict,
                                               bleu_eval,
                                               rouge_eval)
        bleu_rouge['Bleu-4'] = bleu4
        bleu_rouge['Rouge-L'] = rouge_l
    else:
        bleu_rouge = None

    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/loss".format(data_type), simple_value=avg_loss), ])
    bleu_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/f1".format(data_type), simple_value=bleu4), ])
    rouge_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/em".format(data_type), simple_value=rouge_l), ])
    return avg_loss, bleu_rouge, [loss_sum, bleu_sum, rouge_sum]


def find_best_answer_for_passage(start_probs, end_probs, passage_len=None, max_a_len=None):
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
        for ans_len in range(max_a_len):
            end_idx = start_idx + ans_len
            if end_idx >= passage_len:
                continue
            prob = start_probs[start_idx] * end_probs[end_idx]
            if prob > max_prob:
                best_start = start_idx
                best_end = end_idx
                max_prob = prob
    return [best_start, best_end], max_prob
