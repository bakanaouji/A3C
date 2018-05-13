import tensorflow as tf
import numpy as np

from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Input


class AtariModel(object):
    def __init__(self, num_actions, frame_width, frame_height):
        # 入力
        self.s = tf.placeholder(
            tf.float32, [None, 4, frame_width, frame_height]
        )
        inputs = Input(shape=(4, frame_width, frame_height))
        # 共通の中間層
        shared = Conv2D(
            16, (8, 8), strides=(4, 4), activation='relu',
            input_shape=(4, frame_width, frame_height),
            data_format='channels_first'
        )(inputs)
        shared = Conv2D(32, (4, 4), strides=(2, 2), activation='relu',
                        data_format='channels_first')(shared)
        shared = Flatten()(shared)
        shared = Dense(256, activation='relu')(shared)

        # 政策
        action_probs = Dense(num_actions, activation='softmax')(shared)

        # 価値関数
        state_value = Dense(1)(shared)

        self.model = Model(inputs=inputs,
                           outputs=[action_probs, state_value])
        self.p_out, self.v_out = self.model(self.s)
        self.weights = self.model.trainable_weights

    def take_action(self, sess, observation):
        action_p = self.p_out.eval(session=sess,
                                   feed_dict={self.s: observation})
        return np.random.choice(action_p[0].size, 1,
                                p=action_p[0])[0]

    def estimate_value(self, sess, observation):
        return self.v_out.eval(session=sess,
                               feed_dict={self.s: observation})[0][0]
