import tensorflow as tf

from utils.scheduler import Scheduler


class GlobalServer:
    def __init__(self, model, args):
        # initialize model
        self.model = model

        self.weights = self.model.model.trainable_weights

        self.lr = tf.placeholder(tf.float32, [])
        self.scheduler = Scheduler(args.learn_rate, args.tmax)

        # define optimizer
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr,
                                                   decay=args.decay,
                                                   epsilon=1e-5)
