import tensorflow as tf
import pickle as pkl
import os
from tqdm import tqdm
import ujson as json
from collections import Counter
import numpy as np


def process_file(filenames, data_type, max_p_len=500):
    print("Generating {} examples...".format(data_type))
    is_train = False
    if data_type == 'train':
        is_train = True
    examples = []
    eval_examples = {}
    total = 0
    for filename in filenames:
        with open(filename, 'r', encoding='utf8') as fh:
            for line in fh:
                source = json.loads(line.strip())
                if is_train:
                    if len(source['answer_spans']) == 0:
                        continue
                    if source['answer_spans'][0][1] >= max_p_len:
                        continue
                total += 1
                answers = []
                if 'answer_docs' in source:
                    del source['fake_answers']
                    del source['segmented_answers']
                    answers = source['answers']
                question_tokens = source['segmented_question']
                passages = []
                passages_len = []
                start, end, answer_passages = 0, 0, 0
                if 'answer_docs' in source and len(source['answer_docs']):
                    start = source['answer_spans'][0][0]
                    end = source['answer_spans'][0][1]
                    answer_passages = source['answer_docs'][0]
                for idx, doc in enumerate(source['documents']):
                    del doc['paragraphs']
                    para_len = 0
                    if is_train:
                        para_len = min(len(doc['segmented_paragraphs'][doc['most_related_para']]), max_p_len)
                        passages += doc['segmented_paragraphs'][doc['most_related_para']][:para_len]
                    else:
                        para_infos = []
                        for para_tokens in doc['segmented_paragraphs']:
                            # para_tokens 每篇文档分词后的段落，question_tokens 问题分词
                            common_with_question = Counter(para_tokens) & Counter(question_tokens)
                            correct_preds = sum(common_with_question.values())
                            if correct_preds == 0:
                                recall_wrt_question = 0
                            else:
                                recall_wrt_question = float(correct_preds) / len(question_tokens)
                            para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
                        # 排序 选出与question匹配recall最高的para_tokens
                        para_infos.sort(key=lambda x: (-x[1], x[2]))
                        fake_passage_tokens = []
                        for para_info in para_infos[:1]:
                            fake_passage_tokens += para_info[0]
                        para_len = min(len(fake_passage_tokens), max_p_len)
                        passages += fake_passage_tokens[:para_len]
                    if idx < answer_passages:
                        start += para_len
                        end += para_len
                    passages_len.append(para_len)
                example = {'passages': passages,
                           'question_tokens': question_tokens,
                           'answer_passages': answer_passages,
                           'start_id': start,
                           'end_id': end,
                           'id': total}
                if not is_train:
                    eval_examples[str(total)] = {'passages': passages,
                                                 'passages_len': passages_len,
                                                 'answers': answers,
                                                 'answer_passages': answer_passages,
                                                 'question': source['segmented_question'],
                                                 'question_id': source['question_id'],
                                                 'question_type': source['question_type']}
                examples.append(example)
    # random.shuffle(examples)
    print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def get_embedding(data_type, emb_file=None, vec_size=None, token2id_dict=None):
    print("Generating {} embedding...".format(data_type))
    filtered_tokens = {}
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, 'rb') as fin:
            trained_embeddings = pkl.load(fin)
        fin.close()
        filtered_tokens = trained_embeddings.keys()

    NULL = "<NULL>"
    OOV = "<OOV>"
    # token2id
    token2id = {token: idx for idx, token in
                enumerate(filtered_tokens, 2)} if token2id_dict is None else token2id_dict
    id2token = {idx: token for idx, token in enumerate(filtered_tokens, 2)}
    token2id[NULL] = 0
    token2id[OOV] = 1
    id2token['0'] = NULL
    id2token['1'] = OOV
    embedding_mat = np.zeros([len(token2id), vec_size])
    # idx2emb = {idx: embedding_mat[token] for token, idx in token2id.items()}
    # embedding_mat = [idx2emb[idx] for idx in range(len(idx2emb))]
    for token in filtered_tokens:
        # if token in trained_embeddings:
        embedding_mat[token2id[token]] = trained_embeddings[token]
    return embedding_mat, token2id, id2token


def build_features(config, examples, data_type, out_file, word2id):
    para_limit = config.max_p_len
    ques_limit = config.max_q_len

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    meta = {}
    for example in tqdm(examples):
        total += 1
        passages_token_ids = np.zeros([config.max_p_num * para_limit], dtype=np.int32)
        question_token_ids = np.zeros([ques_limit], dtype=np.int32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2id:
                    return word2id[each]
            return 1

        # passages token转id
        idx = 0
        for pdx, passage_token in enumerate(example['passages']):
            passages_token_ids[pdx] = _get_word(passage_token)
        # 问题token转id
        question_token_num = min(len(example['question_tokens']), ques_limit)
        for i in range(question_token_num):
            question_token_ids[i] = _get_word(example['question_tokens'][i])

        record = tf.train.Example(features=tf.train.Features(feature={
            "passages_token_ids": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[passages_token_ids.tostring()])),
            "question_token_ids": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[question_token_ids.tostring()])),
            "start_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example['start_id']])),
            "end_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example['end_id']])),
            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
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

    train_examples, train_eval = process_file(config.train_files, "train", config.max_p_len)
    # 创建train TFRecord文件
    train_meta = build_features(config, train_examples, "train", flags.train_record_file, token2id)
    save(flags.train_eval_file, train_eval, message="train eval")
    save(flags.train_meta, train_meta, message="dev meta")
    del train_examples, train_eval, train_meta

    dev_examples, dev_eval = process_file(config.dev_files, "dev", config.max_p_len)
    # 创建dev TFRecord文件
    dev_meta = build_features(config, dev_examples, "dev", flags.dev_record_file, token2id)
    save(flags.dev_eval_file, dev_eval, message="dev eval")
    save(flags.dev_meta, dev_meta, message="dev meta")
    del dev_examples, dev_eval, dev_meta

    test_examples, test_eval = process_file(config.test_files, "test", config.max_p_len)
    # # 创建test TFRecord文件
    test_meta = build_features(config, test_examples, "test", flags.test_record_file, token2id)
    save(flags.test_eval_file, test_eval, message="test eval")
    save(flags.test_meta, test_meta, message="test meta")
    del test_examples, test_eval, test_meta

    # save(flags.token2id_file, token2id, message="word2idx")

# def draw_hist(x, bins, label):
#     plt.hist(x=x, bins=bins)
#     plt.xlabel(label)
#     plt.ylabel('Num')
#     plt.show()
