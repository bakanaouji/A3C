import tensorflow as tf


class GlobalServer:
    def __init__(self, model, args):
        # initialize model
        self.model = model

        self.weights = self.model.model.trainable_weights

        # define optimizer
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-3,
                                                   decay=args.decay,
                                                   epsilon=1e-5)
