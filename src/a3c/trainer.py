import threading
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

        self.worker_num = args.worker_num

        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    def train(self):
        workers = [Worker(i) for i in range(self.worker_num)]
        threads = [threading.Thread(target=workers[i].train,
                                    args=(self.env_name, self.seed))
                   for i in range(len(workers))]
        for thread in threads:
            thread.start()
