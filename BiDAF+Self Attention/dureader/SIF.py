import pickle as pkl
import os
import sys
import ujson as json
import logging
import numpy as np
from gensim.models import word2vec
from sklearn.decomposition import TruncatedSVD


class SIFModel(object):
    def __init__(self, args, logger, pre_train, train_files=[], dev_files=[], test_files=[], a=1e-3, embed_dim=300):
        self.logger = logger
        self.segmented_dir = args.segmented_dir
        self.prepared_dir = args.prepared_dir
        self.a = a
        self.embed_dim = embed_dim
        self.weighted_word_dict = None
        self.pc = None
        self.train_set, self.dev_set, self.test_set = [], [], []

        if pre_train:
            for train_file in train_files:
                self.train_set += self._load_dataset(train_file, train=True)
            self.train_set_seg = os.path.join(self.segmented_dir, 'train_set.seg')
            self.logger.info('Writing train_set.seg')
            self._write_data(self.train_set, self.train_set_seg)
            del self.train_set

            for dev_file in dev_files:
                self.dev_set += self._load_dataset(dev_file)
            self.dev_set_seg = os.path.join(self.segmented_dir, 'dev_set.seg')
            self.logger.info('Writing dev_set.seg')
            self._write_data(self.dev_set, self.dev_set_seg)
            del self.dev_set

            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            self.test_set_seg = os.path.join(self.segmented_dir, 'test_set.seg')
            self.logger.info('Writing test_set.seg')
            self._write_data(self.test_set, self.test_set_seg)
            del self.test_set

    def _load_dataset(self, data_path, train=False):
        fin = open(data_path, 'r', encoding='utf8')
        data_set = []
        for lidx, line in enumerate(fin):
            sample = json.loads(line.strip())
            del sample['question']
            if train:
                del sample['answers']
                del sample['fake_answers']
                del sample['segmented_answers']
            sample['passages'] = []
            for d_idx, doc in enumerate(sample['documents']):
                if train:
                    most_related_para = doc['most_related_para']
                    sample['passages'].append({'passage_tokens': doc['segmented_paragraphs'][most_related_para]})
                else:
                    for segmented_paragraph in doc['segmented_paragraphs']:
                        sample['passages'].append({'passage_tokens': segmented_paragraph})
            del sample['documents']
            data_set.append(sample)
        fin.close()
        return data_set

    def _write_data(self, data_set, tar_dir):
        with open(tar_dir, 'w', encoding='utf8') as f:
            for sample in data_set:
                f.write(' '.join(sample['segmented_question']) + '\n')
                for passage in sample['passages']:
                    f.write(' '.join(passage['passage_tokens']) + '\n')
                del sample
        f.close()

    def train_embeddings(self):
        sys.path.append('..')
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
        logging.root.setLevel(level=logging.INFO)
        self.logger.info("running %s" % ' '.join(sys.argv))

        model = word2vec.Word2Vec(word2vec.PathLineSentences(self.segmented_dir), size=300, min_count=2, workers=8,
                                  iter=15)
        w2v_dict = {}
        for word in model.wv.vocab:
            w2v_dict[word] = model[word]
        with open(os.path.join(self.prepared_dir, 'w2v_dic.pkl'), 'wb') as f:
            pkl.dump(w2v_dict, f)
        f.close()
        model.wv.save_word2vec_format(os.path.join(self.prepared_dir, 'w2v_model.bin'), binary=True)

    def get_dict_word_fre(self):
        word_all_num = 0
        dict_word_num = {}
        dict_word_fre = {}
        for root, dirs, files in os.walk(self.segmented_dir):
            for file_name in files:
                with open(os.path.join(self.segmented_dir, file_name), 'r', encoding='utf8') as f:
                    for line in f.readlines():
                        line = line.replace('\n', '')
                        words = line.split(' ')
                        for word in words:
                            word_all_num += 1
                            if word in dict_word_num:
                                dict_word_num[word] += 1
                            else:
                                dict_word_num[word] = 1
                f.close()
        for word in dict_word_num:
            dict_word_fre[word] = dict_word_num[word] / word_all_num
        return word_all_num, dict_word_fre

    def get_dict_word_weight(self):
        word_all_num, dict_word_fre = self.get_dict_word_fre()
        self.logger.info('Total words num is {}'.format(word_all_num))
        if self.a <= 0:
            self.a = 1.0
        dict_word_weight = {}
        for word in dict_word_fre:
            dict_word_weight[word] = self.a / (self.a + dict_word_fre[word])
        return dict_word_weight

    def load_model(self):
        with open(os.path.join(self.prepared_dir, 'weighted_word_dict.pkl'), 'rb') as fww:
            self.weighted_word_dict = pkl.load(fww)
        with open(os.path.join(self.prepared_dir, 'pc.pkl'), 'rb') as fpc:
            self.pc = pkl.load(fpc)

    def get_weighted_embedding(self, sentence):
        # init the sentence embedding
        weighted_embedding = np.array([0.0] * self.embed_dim)
        for word in sentence:
            # weighted_embedding += self.weighted_word_dict[word]
            if word in self.weighted_word_dict:
                weighted_embedding += self.weighted_word_dict[word]
            else:
                weighted_embedding += np.array([1.0] * self.embed_dim) * 0.001
        return weighted_embedding

    def get_weighted_embedding_list(self, dict_word_weight):
        weighted_embedding_list = []
        weighted_word_dict = {}
        with open(os.path.join(self.prepared_dir, 'w2v_dic.pkl'), 'rb') as fin:
            w2v_model = pkl.load(fin)
        fin.close()
        for root, dirs, files in os.walk(self.segmented_dir):
            for file_name in files:
                with open(os.path.join(self.segmented_dir, file_name), 'r', encoding='utf8') as f:
                    for line in f.readlines():
                        line = line.replace('\n', '')
                        words = line.split(' ')
                        weighted_embedding = np.array([0.0] * self.embed_dim)
                        for word in words:
                            if word not in weighted_word_dict:
                                if word in w2v_model:
                                    weighted_word_embedding = w2v_model[word] * dict_word_weight[word]
                                else:
                                    weighted_word_embedding = np.array([1.0] * self.embed_dim) * 0.001
                                weighted_word_dict[word] = weighted_word_embedding
                            weighted_embedding += weighted_word_dict[word]
                        weighted_embedding_list.append(weighted_embedding)
                f.close()
        pkl.dump(weighted_word_dict, open(os.path.join(self.prepared_dir, 'weighted_word_dict.pkl'), 'wb'))
        return np.array(weighted_embedding_list)

    def compute_pc(self, x, npc=1):
        svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
        svd.fit(x)
        return svd.components_

    def remove_pc(self, x, npc=1):
        """
        Remove the projection on the principal components
        :param x: x[i,:] is a data point
        :param npc: number of principal components to remove
        :return: XX[i, :] is the data point after removing its projection
        """
        pc = self.compute_pc(x, npc)
        if npc == 1:
            xx = x - x.dot(pc.transpose()) * pc
        else:
            xx = x - x.dot(pc.transpose()).dot(pc)
        return xx

    def build_pc_and_sif_embedding_list(self):
        dict_word_weight = self.get_dict_word_weight()
        # pkl.dump(dict_word_weight, open(os.path.join(self.prepared_dir, 'dict_word_weight.pkl'), 'wb'))
        weighted_embedding_list = self.get_weighted_embedding_list(dict_word_weight)
        self.logger.info('Finish building the weighted embedding list of sentence list')
        pc = self.compute_pc(weighted_embedding_list)
        pkl.dump(pc, open(os.path.join(self.prepared_dir, 'pc.pkl'), 'wb'))
        self.logger.info('Finish building the pc')
        # sif_embedding_list = self.remove_pc(weighted_embedding_list)
        # pickle.dump(sif_embedding_list, open(params.dump_sif_embedding_list_path, 'wb'))
        # self.logger.info('Finish building the sif_embedding')

    def get_sif_embedding(self, text):
        sentence_embedding = self.get_weighted_embedding(text)
        rmpc_sentence_embedding = sentence_embedding - sentence_embedding.dot(self.pc.transpose()) * self.pc
        return rmpc_sentence_embedding
