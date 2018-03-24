import sys
import os
import logging
from gensim.models import word2vec
from .json_to_sentence import load_data


def pre_train(brc_data, segmented_dir):
    # parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    # path_settings = parser.add_argument_group('path settings')
    # path_settings.add_argument('--train_files', nargs='+',
    #                            default=['../data/trainset/search.train.json'],
    #                            help='list of files that contain the preprocessed train data')
    # path_settings.add_argument('--dev_files', nargs='+',
    #                            default=['../data/devset/search.dev.json'],
    #                            help='list of files that contain the preprocessed dev data')
    # path_settings.add_argument('--test_files', nargs='+',
    #                            default=['../data/testset/search.test.json'],
    #                            help='list of files that contain the preprocessed test data')
    # path_settings.add_argument('--segmented_dir', default='../data/segmented',
    #                            help='the dir to store segmented sentences')

    sys.path.append('..')
    # args = parser.parse_args()
    # for files in args.train_files + args.dev_files + args.test_files:
    #     json_to_sentence.load_data(files, args.segmented_dir)
    load_data(brc_data, segmented_dir)

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
