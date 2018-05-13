import tensorflow as tf
import numpy as np

from keras.models import Model
from keras.layers import Dense, Input


class NormalModel(object):
    def __init__(self, num_actions, num_states):
        # 入力
        self.s = tf.placeholder(
            tf.float32, [None, num_states]
        )
        inputs = Input(shape=(num_states,))
        shared = Dense(16, activation='relu')(inputs)

        # 政策
        action_probs = Dense(num_actions, activation='softmax')(shared)

        # 価値関数
        state_value = Dense(1)(shared)

        self.model = Model(inputs=inputs,
                           outputs=[action_probs, state_value])
        self.p_out, self.v_out = self.model(self.s)
        self.operation = None

    def take_action(self, sess, observation):
        action_p = self.p_out.eval(session=sess,
                                   feed_dict={self.s: observation})
        return np.random.choice(action_p[0].size, 1,
                                p=action_p[0])[0]

    def estimate_value(self, sess, observation):
        return self.v_out.eval(session=sess,
                               feed_dict={self.s: observation})[0][0]

    def update_param(self, sess, src_weights):
        if self.operation is None:
            weights = self.model.trainable_weights
            self.operation = [src_weight.assign(weight) for weight, src_weight
                              in zip(src_weights, weights)]
        sess.run(self.operation)
