import tensorflow as tf

from utils.scheduler import Scheduler
from utils.summarizer import Summarizer


class GlobalServer:
    def __init__(self, model, args):
        """
        全てのスレッドで共有するパラメータを管理するサーバー．
        :param model:   モデル
        :param args:    パラメータ群
        """
        # global shared parameter vectors
        self.weights = model.model.trainable_weights

        # RMSPropの学習率
        self.scheduler = Scheduler(args.learn_rate, args.tmax)
        self.lr = tf.placeholder(tf.float32, [])

        # optimizer
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr,
                                                   decay=args.decay,
                                                   epsilon=0.1)

        # summarizer
        self.summarizer = Summarizer('../data/summaries')
