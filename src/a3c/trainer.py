import tensorflow as tf
import numpy as np
import random

from multiprocessing import Value, Process

from a3c.worker import Worker


class Trainer(object):
    def __init__(self, args):
        self.env_name = args.env_name
        self.seed = args.seed
        self.width = args.width
        self.height = args.height

        self.tmax = args.tmax
        self.worker_num = args.worker_num

        # 乱数シードセット
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    def train(self):
        # ワーカーとスレッド初期化
        global_step = Value('i', self.tmax)
        workers = [Worker(i, global_step) for i in range(self.worker_num)]
        processes = [Process(target=workers[i].train,
                             args=(self.env_name, self.seed))
                     for i in range(len(workers))]

        # 各スレッドで学習実行開始
        for thread in processes:
            thread.start()
