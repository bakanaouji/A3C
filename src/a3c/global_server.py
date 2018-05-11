import tensorflow as tf

from models.a3c_lstm import A3CLSTM


class GlobalServer:
    def __init__(self, action_n, history_len, width, height):
        # initialize model
        self.model = A3CLSTM(action_n, history_len, width, height)

        self.weights = self.model.model.trainable_weights

        # define optimizer
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-3,
                                                   decay=0.99,
                                                   epsilon=1e-5)
