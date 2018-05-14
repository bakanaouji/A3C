from threading import Thread

import tensorflow as tf
import numpy as np
import random

from a3c.global_server import GlobalServer
from a3c.worker import Worker


class Trainer(object):
    def __init__(self, args, envs, models):
        """
        学習を行うTrainerクラス．
        :param args:    パラメータ群
        :param envs:    環境
        :param models:  モデル
        """
        self.args = args
        self.envs = envs
        self.models = models

        # 乱数シードセット
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    def train(self):
        """
        学習を実行
        """
        # Sessionの構築
        sess = tf.InteractiveSession()

        # global shared parameterを初期化
        global_server = GlobalServer(self.models[len(self.models) - 1],
                                     self.args)

        # ワーカーとスレッド初期化
        workers = [Worker(sess, global_server, self.envs[i], self.models[i], i,
                          self.args)
                   for i in range(self.args.worker_num)]
        threads = [Thread(target=worker.train, args=()) for worker in workers]

        # 変数の初期化
        sess.run(tf.global_variables_initializer())

        # 各スレッドで学習実行開始
        for thread in threads:
            thread.start()

        # while True:
        #     for worker in workers:
        #         worker.env.render()
