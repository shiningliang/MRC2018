import tensorflow as tf
import pickle as pkl
from tqdm import tqdm
import ujson as json
import numpy as np
import jieba
import os

TYPE = {'Yes': 0, 'No': 1, 'Depends': 2, 'No_Opinion': 1}


def split_answers(answers):
    tokens = jieba.cut(answers)
    return [token for token in tokens]


def filter_questions(filenames):
    questions = {}
    for filename in filenames:
        with open(filename, 'r', encoding='utf8') as fh:
            for line in fh:
                source = json.loads(line.strip())
                if source['question_type'] != 'YES_NO':
                    continue
                questions[source['question_id']] = source['segmented_question']
    print("{} questions in total".format(len(questions)))
    return questions


def process_test_file(filename, questions, max_p_len=500):
    print("Generating test examples...")
    total = 0
    examples = []
    other_examples = []
    eval_examples = {}
    with open(filename, 'r', encoding='utf8') as fh:
        for line in fh:
            source = json.loads(line.strip())
            if source['question_type'] != 'YES_NO':
                other_examples.append(source)
                continue
            total += 1
            answer_type = -1
            example = {'question_tokens': questions[str(source['question_id'])],
                       'answer_tokens': split_answers(source['answers'][0]),
                       'answer_type': answer_type,
                       'id': total}
            eval_examples[str(total)] = {'question_id': source['question_id'],
                                         'answers': source['answers']}
            examples.append(example)
    # random.shuffle(examples)
    print("{} questions in total".format(len(examples)))
    return examples, eval_examples, other_examples


def process_file(filenames, data_type, max_p_len=500):
    print("Generating {} examples...".format(data_type))
    total = 0
    examples = []
    eval_examples = {}
    for filename in filenames:
        with open(filename, 'r', encoding='utf8') as fh:
            for line in fh:
                source = json.loads(line.strip())
                if source['question_type'] != 'YES_NO':
                    continue
                if len(source['answer_spans']) == 0:
                    continue
                if source['answer_spans'][0][1] >= max_p_len:
                    continue
                question_tokens = source['segmented_question']
                for idx, answer_tokens in enumerate(source['segmented_answers']):
                    total += 1
                    answer_type = TYPE[source['yesno_answers'][idx]] if len(source['yesno_answers']) else -1
                    example = {'question_tokens': question_tokens,
                               'answer_tokens': answer_tokens,
                               'answer_type': answer_type,
                               'id': total}
                    eval_examples[str(total)] = {'question_id': source['question_id'],
                                                 'answer_type': answer_type}
                    examples.append(example)
    # random.shuffle(examples)
    print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def build_features(config, examples, data_type, out_file, word2id):
    ans_limit = config.max_a_len
    ques_limit = config.max_q_len

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    meta = {}
    for example in tqdm(examples):
        total += 1
        answer_token_ids = np.zeros([ans_limit], dtype=np.int32)
        question_token_ids = np.zeros([ques_limit], dtype=np.int32)
        answer_type = np.zeros([3], dtype=np.int32)
        answer_type[example['answer_type']] = 1

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2id:
                    return word2id[each]
            return 1

        answers_token_num = min(len(example['answer_tokens']), ques_limit)
        for i in range(answers_token_num):
            answer_token_ids[i] = _get_word(example['answer_tokens'][i])
        question_token_num = min(len(example['question_tokens']), ques_limit)
        for j in range(question_token_num):
            question_token_ids[j] = _get_word(example['question_tokens'][j])

        record = tf.train.Example(features=tf.train.Features(
            feature={
                'answer_token_ids': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[answer_token_ids.tostring()])),
                'question_token_ids': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[question_token_ids.tostring()])),
                'answer_type': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[answer_type.tostring()])),
                'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[example['id']]))
            }))
        writer.write(record.SerializeToString())
    print("Build {} instances of features in total".format(total))
    meta["total"] = total
    writer.close()
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def prepro(config, flags):
    token2id = None
    if os.path.isfile(flags.token2id_file):
        with open(flags.token2id_file, 'r') as fh:
            token2id = json.load(fh)
    # train_examples, _ = process_file(config.train_files, 'train')
    # train_meta = build_features(config, train_examples, 'train', flags.train_record_file, token2id)
    # save(flags.train_meta, train_meta, message='train meta')
    # del train_examples, train_meta
    #
    # dev_examples, dev_eval = process_file(config.dev_files, "dev")
    # # 创建dev TFRecord文件
    # dev_meta = build_features(config, dev_examples, "dev", flags.dev_record_file, token2id)
    # save(flags.dev_eval_file, dev_eval, message="dev eval")
    # save(flags.dev_meta, dev_meta, message="dev meta")
    # del dev_examples, dev_eval, dev_meta

    # filtered_questions = filter_questions(config.test_files)
    # save(flags.filtered_questions, filtered_questions, message='filtered questions')
    filtered_questions = None
    if os.path.isfile(flags.token2id_file):
        with open(flags.filtered_questions, 'r') as fh:
            filtered_questions = json.load(fh)
    test_examples, test_eval, other_examples = process_test_file(flags.predicted_answers, filtered_questions)
    # 创建test TFRecord文件
    test_meta = build_features(config, test_examples, "test", flags.test_record_file, token2id)
    save(flags.test_eval_file, test_eval, message="test eval")
    save(flags.final_file, other_examples, message="test final")
    with open(flags.final_file, 'w') as fout:
        for example in other_examples:
            fout.write(json.dumps(example, ensure_ascii=False) + '\n')
    fout.close()
    save(flags.test_meta, test_meta, message="test meta")
    del test_examples, test_meta, test_eval, other_examples
