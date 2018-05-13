from threading import Thread

import tensorflow as tf
import numpy as np
import random

from a3c.global_server import GlobalServer
from a3c.worker import Worker


class Trainer(object):
    def __init__(self, args, envs, models):
        self.envs = envs
        self.models = models

        self.args = args

        # 乱数シードセット
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    def train(self):
        # initialize session
        sess = tf.InteractiveSession()

        # initialize global shared parameter
        global_server = GlobalServer(self.models[len(self.models) - 1],
                                     self.args)

        # ワーカーとスレッド初期化
        workers = [Worker(sess, global_server, self.envs[i], self.models[i], i,
                          self.args)
                   for i in range(self.args.worker_num)]
        threads = [Thread(target=worker.train, args=()) for worker in workers]

        # initialize variables
        sess.run(tf.global_variables_initializer())

        # 各スレッドで学習実行開始
        for thread in threads:
            thread.start()

        # while True:
        #     for worker in workers:
        #         worker.env.render()
