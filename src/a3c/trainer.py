from threading import Thread

import tensorflow as tf
import numpy as np
import random

from a3c.worker import Worker
from envs.env_wrappers import make_atari, wrap_deepmind
from models.a3c_lstm import A3CLSTM


class Trainer(object):
    def __init__(self, args):
        self.env_name = args.env_name
        self.seed = args.seed
        self.width = args.width
        self.height = args.height

        self.tmax = args.tmax
        self.batch_size = args.batch_size
        self.worker_num = args.worker_num
        self.history_len = args.history_len
        self.discount_fact = args.discount_fact

        # 乱数シードセット
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    def train(self):
        # initialize model
        env = make_atari(self.env_name)
        env = wrap_deepmind(env)
        a3c_lstm = A3CLSTM(env.action_space.n, self.history_len, self.width,
                           self.height)
        env.close()

        # initialize session
        sess = tf.InteractiveSession()

        # ワーカーとスレッド初期化
        workers = [Worker(a3c_lstm, sess, self.env_name, i, self.seed,
                          self.tmax, self.batch_size, self.discount_fact,
                          self.history_len, self.width, self.height)
                   for i in range(self.worker_num)]
        thread = [Thread(target=workers[i].train,
                         args=())
                  for i in range(len(workers))]

        # initialize variables
        sess.run(tf.global_variables_initializer())

        # 各スレッドで学習実行開始
        for thread in thread:
            thread.start()

        while True:
            for worker in workers:
                worker.env.render()
