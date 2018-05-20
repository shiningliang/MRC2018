import sys
import os
import logging
# import argparse
from gensim.models import word2vec


def pre_train(segmented_dir):
    sys.path.append('..')

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    model = word2vec.Word2Vec(word2vec.PathLineSentences(segmented_dir), size=300, min_count=2, workers=8, iter=10)
    with open(os.path.join(segmented_dir, 'w2v_dic.data'), 'w', encoding='utf-8') as f:
        for word in model.wv.vocab:
            f.write(word + ' ')
            f.write(' '.join(list(map(str, model[word]))))
            f.write('\n')
    f.close()

    model.save_word2vec_format(os.path.join(segmented_dir, 'w2v_model.bin'), binary=True)


def write_data(data_set, tar_dir):

    with open(tar_dir, 'w', encoding='utf8') as f:
        for sample in data_set:
            f.write(' '.join(sample['segmented_question']) + '\n')
            for passage in sample['passages']:
                f.write(' '.join(passage['passage_tokens']) + '\n')
            del sample
    f.close()
