from threading import Thread

import tensorflow as tf
import numpy as np
import random

from a3c.worker import Worker


class Trainer(object):
    def __init__(self, args):
        self.env_name = args.env_name
        self.seed = args.seed
        self.width = args.width
        self.height = args.height

        self.tmax = args.tmax
        self.batch_size = args.batch_size
        self.worker_num = args.worker_num

        # 乱数シードセット
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    def train(self):
        # ワーカーとスレッド初期化
        workers = [
            Worker(self.env_name, i, self.seed, self.tmax, self.batch_size)
            for i in range(self.worker_num)]
        thread = [Thread(target=workers[i].train,
                         args=())
                  for i in range(len(workers))]

        # 各スレッドで学習実行開始
        for thread in thread:
            thread.start()

        while True:
            for worker in workers:
                worker.env.render()
