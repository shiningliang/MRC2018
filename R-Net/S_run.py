import tensorflow as tf

# sys.path.append('..')
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import ujson as json
import argparse
import logging
from S_prepro import prepro
from S_model import Model
from S_util import get_record_parser, evaluate_batch, get_batch_dataset, get_dataset
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='tensorflow')


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
    train_settings.add_argument('--optim', default='adadelta',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.5,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=0.7,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=8,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=64,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--layer_num', type=int, default=1,
                                help='num of LSTM layers')
    model_settings.add_argument('--max_p_num', type=int, default=5,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=200,
                                help='max length of answer')
    model_settings.add_argument('--num_threads', type=int, default=8,
                                help='Number of threads in input pipeline')
    model_settings.add_argument('--capacity', type=int, default=260000,
                                help='Batch size of data set shuffle')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['data/demo/trainset/search.train.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['data/demo/devset/search.dev.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['data/demo/testset/search.test.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--prepared_dir', default='data/S/prepared',
                               help='the dir to store prepared ')
    path_settings.add_argument('--brc_dir', default='data/baidu',
                               help='the dir with preprocessed baidu reading comprehension data')
    path_settings.add_argument('--model_dir', default='data/S/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='data/S/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='data/S/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()


def modern_train(config, file_paths):
    logger = logging.getLogger('brc')
    # 用w2c_dict.pkl代替
    logger.info('Loading token embeddings...')
    with open(file_paths.token_emb_file, 'rb') as fh:
        token_embeddings = np.array(json.load(fh), dtype=np.float32)
    logger.info('Loading dev eval file...')
    with open(file_paths.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    logger.info('Loading train meta...')
    with open(file_paths.train_meta, "r") as fh:
        train_meta = json.load(fh)
    logger.info('Loading dev meta...')
    with open(file_paths.dev_meta, "r") as fh:
        dev_meta = json.load(fh)
    train_total = train_meta['total']
    dev_total = dev_meta['total']

    # 返回解析单个样本的函数
    parser = get_record_parser(config)
    train_dataset = get_batch_dataset(file_paths.train_record_file, parser, config)
    dev_dataset = get_dataset(file_paths.dev_record_file, parser, config)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()
    logger.info('Building model...')
    model = Model(config, iterator, token_embeddings)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        writer = tf.summary.FileWriter(config.summary_dir)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())

        max_rouge = 0
        min_loss = 100000
        log_every_n_batch, n_batch_loss = 50, 0
        for epoch in range(config.epochs):
            logger.info('Training the model for epoch {}'.format(epoch + 1))
            train_loss = []
            lr_decay = 0.9 ** max(epoch - 5, 0)
            sess.run(tf.assign(model.lr, tf.constant(config.learning_rate * lr_decay, dtype=tf.float32)))
            sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
            # train
            for i in range(train_total // config.batch_size + 1):
                loss, train_op = sess.run([model.loss, model.train_op], feed_dict={handle: train_handle})
                bitx = i + 1
                n_batch_loss += loss
                # logger.info('epoch-{} batch-{} loss-{}'.format(epoch+1, bitx, loss))
                if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                    logger.info('Average loss from batch {} to {} is {}'.format(
                        bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                    n_batch_loss = 0
                train_loss.append(loss)
            train_loss = np.mean(train_loss)
            logger.info('Average train loss for epoch {} is {}'.format(epoch + 1, train_loss))
            loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=train_loss), ])
            writer.add_summary(loss_sum, epoch)
            # eval
            sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
            logger.info('Evaluating the model after epoch {}'.format(epoch + 1))
            eval_loss, bleu_rouge, summ = evaluate_batch(model, dev_total // config.batch_size, dev_eval_file,
                                                         sess, 'dev', handle, dev_handle, config, logger)

            logger.info('Dev eval loss {}'.format(eval_loss))
            logger.info('Dev eval result: {}'.format(bleu_rouge))
            for s in summ:
                writer.add_summary(s, epoch)
            writer.flush()
            filename = os.path.join(config.model_dir, "model_{}.ckpt".format(epoch + 1))
            if eval_loss < min_loss:
                # max_rouge = bleu_rouge['Rouge-L']
                min_loss = eval_loss
                saver.save(sess, filename)


def modern_predict(config, file_paths):
    logger = logging.getLogger('brc')
    # 用w2c_dict.pkl代替
    logger.info('Loading token embeddings...')
    with open(file_paths.token_emb_file, 'rb') as fh:
        token_embeddings = np.array(json.load(fh), dtype=np.float32)
    logger.info('Loading test eval file...')
    with open(file_paths.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    logger.info('Loading test meta...')
    with open(file_paths.test_meta, "r") as fh:
        meta = json.load(fh)
    total = meta["total"]
    test_batch = get_dataset(file_paths.test_record_file, get_record_parser(config), config).make_one_shot_iterator()
    logger.info('Building model...')
    model = Model(config, test_batch, token_embeddings, trainable=False)
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(config.model_dir))
        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
        evaluate_batch(model, total // config.batch_size, eval_file, sess, 'predict', None, None, config, logger,
                       result_prefix='test.predicted')


def run():
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
    # model_saver = None

    logger = logging.getLogger("brc")
    logger.info('Checking the data files...')
    # 检查数据文件是否存在
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    logger.info('Preparing the directories...')
    for dir_path in [args.prepared_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    class FilePaths(object):
        def __init__(self):
            # 运行记录文件
            self.train_record_file = os.path.join(args.prepared_dir, 'train.tfrecords')
            self.dev_record_file = os.path.join(args.prepared_dir, 'dev.tfrecords')
            # TODO 生成test1 test2
            self.test_record_file = os.path.join(args.prepared_dir, 'test.tfrecords')
            # embedding 文件
            self.token2id_file = os.path.join(args.prepared_dir, 'token2id.json')
            self.token_emb_file = os.path.join(args.prepared_dir, 'token_emb.json')
            # 评估文件
            self.train_eval_file = os.path.join(args.prepared_dir, "train_eval.json")
            self.dev_eval_file = os.path.join(args.prepared_dir, "dev_eval.json")
            # TODO 生成test1 test2
            self.test_eval_file = os.path.join(args.prepared_dir, "test_eval.json")
            # 计数文件
            self.train_meta = os.path.join(args.prepared_dir, "train_meta.json")
            self.dev_meta = os.path.join(args.prepared_dir, "dev_meta.json")
            self.test_meta = os.path.join(args.prepared_dir, "test_meta.json")
            # index文件
            self.token2id_file = os.path.join(args.prepared_dir, "token2id.json")

    file_paths = FilePaths()
    # 解析各阶段参数，不为空则转到
    if args.prepare:
        prepro(args, file_paths)
    if args.train:
        modern_train(args, file_paths)
    if args.predict:
        modern_predict(args, file_paths)


if __name__ == "__main__":
    run()
