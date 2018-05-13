import tensorflow as tf

from models.atari_model import AtariModel
from models.normal_model import NormalModel


class GlobalServer:
    def __init__(self, num_actions, num_states, history_len, width, height):
        # initialize model
        # self.model = AtariModel(num_actions, history_len, width, height)
        self.model = NormalModel(num_actions, num_states)

        self.weights = self.model.model.trainable_weights

        # define optimizer
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-3,
                                                   decay=0.99,
                                                   epsilon=1e-5)
