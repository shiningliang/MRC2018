import sys
import tensorflow as tf

sys.path.append('..')
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle
import argparse
import logging
from SIF import SIFModel
from dataset import BRCDataset
from vocab import Vocab
from rc_model import RCModel
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings(action='ignore', category=UserWarning, module='h5py')


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--pretrain', action='store_true',
                        help='pretrain word embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=0.7,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=128,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--layer_num', type=int, default=2,
                                help='num of LSTM layers')
    model_settings.add_argument('--max_p_num', type=int, default=5,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=200,
                                help='max length of answer')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['../data/demo/trainset/search.train.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['../data/demo/devset/search.dev.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['../data/demo/testset/search.test.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--segmented_dir', default='../data/segmented',
                               help='the dir to store segmented sentences')
    path_settings.add_argument('--prepared_dir', default='../data/prepared',
                               help='the dir to store prepared ')
    path_settings.add_argument('--brc_dir', default='../data/baidu',
                               help='the dir with preprocessed baidu reading comprehension data')
    path_settings.add_argument('--vocab_dir', default='../data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='../data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='../data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()


def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("brc")
    logger.info('Building SIF Model...')
    sif_model = SIFModel(args, logger, True, args.train_files, args.dev_files, args.test_files)
    logger.info('Training word embeddings...')
    sif_model.train_embeddings()
    logger.info('Building pc and sif embeddings...')
    sif_model.build_pc_and_sif_embedding_list()
    sif_model.load_model()
    logger.info('Building vocabulary...')
    # 构建词典
    vocab = Vocab(lower=True)
    # 载入预训练词向量
    vocab.load_pretrained_embeddings(os.path.join(args.prepared_dir, 'w2v_dic.pkl'))
    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.pkl'), 'wb') as fvocab:
        pickle.dump(vocab, fvocab)
    fvocab.close()
    # 构建数据集
    logger.info('Loading dataset...')
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, args.prepared_dir,
                          args.train_files, args.dev_files, args.test_files, prepare=True)

    logger.info('Done with preparing!')


def train(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("brc")
    logger.info('Loading vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.pkl'), 'rb') as fin:
        vocab = pickle.load(fin)
    fin.close()
    pad_id = vocab.get_id(vocab.pad_token)
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.prepared_dir, args.train_files, args.dev_files, args.test_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    g = tf.Graph()
    with g.as_default():
        rc_model = RCModel(vocab.embeddings, pad_id, args)
        del vocab
        # Train
        with tf.name_scope("Train"):
            logger.info('Training the model...')
            rc_model.train(brc_data, args.epochs, args.batch_size, save_dir=args.result_dir,
                           save_prefix='test.predicted',
                           dropout_keep_prob=args.dropout_keep_prob)
        tf.summary.FileWriter(args.summary_dir, g).close()
        with tf.name_scope('Valid'):
            assert len(args.dev_files) > 0, 'No dev files are provided.'
            logger.info('Evaluating the model on dev set...')
            dev_batches = brc_data.gen_mini_batches('dev', args.batch_size,
                                                    pad_id=pad_id, shuffle=False)
            dev_loss, dev_bleu_rouge = rc_model.evaluate(dev_batches,
                                                         result_dir=args.result_dir,
                                                         result_prefix='dev.predicted')
            logger.info('Loss on dev set: {}'.format(dev_loss))
            logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
            logger.info('Predicted answers are saved to {}'.format(os.path.join(args.result_dir)))
        with tf.name_scope('Test'):
            assert len(args.test_files) > 0, 'No test files are provided.'
            logger.info('Predicting answers for test set...')
            test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                                     pad_id=pad_id, shuffle=False)
            rc_model.evaluate(test_batches,
                              result_dir=args.result_dir, result_prefix='test.predicted')


def run():
    """
    Prepares and runs the whole system.
    """
    # 解析参数
    args = parse_args()

    # 日志命名为brc
    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 是否存储日志
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logger = logging.getLogger("brc")
    logger.info('Checking the data files...')
    # 检查数据文件是否存在
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    logger.info('Preparing the directories...')
    for dir_path in [args.segmented_dir, args.prepared_dir, args.vocab_dir, args.model_dir, args.result_dir,
                     args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 解析各阶段参数，不为空则转到
    if args.prepare:
        prepare(args)
    if args.train:
        train(args)


if __name__ == "__main__":
    run()
