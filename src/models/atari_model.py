import tensorflow as tf
import numpy as np

from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Input


class AtariModel(object):
    def __init__(self, num_actions, frame_width, frame_height):
        """
        Atari用のモデル．
        :param num_actions:     行動数
        :param frame_width:     画面の幅
        :param frame_height:    画面の高さ
        """
        # 入力
        self.s = tf.placeholder(
            tf.float32, [None, frame_width, frame_height, 1]
        )
        s = tf.cast(self.s, tf.float32) / 255.

        # 共通の中間層
        inputs = Input(shape=(frame_width, frame_height, 1))
        shared = Conv2D(
            32, (8, 8), strides=(4, 4), activation='relu',
            input_shape=(frame_width, frame_height, 1),
            data_format='channels_last'
        )(inputs)
        shared = Conv2D(64, (4, 4), strides=(2, 2), activation='relu',
                        data_format='channels_last')(shared)
        shared = Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                        data_format='channels_last')(shared)
        shared = Flatten()(shared)
        shared = Dense(256, activation='relu')(shared)

        # 政策
        action_probs = Dense(num_actions, activation='softmax')(shared)

        # 価値関数
        state_value = Dense(1)(shared)

        # 政策と価値関数の出力
        self.model = Model(inputs=inputs,
                           outputs=[action_probs, state_value])
        self.p_out, self.v_out = self.model(s)

        # 重み
        self.weights = self.model.trainable_weights

    def take_action(self, sess, observation):
        """
        政策に基いて行動を選択
        :param sess:        tensorflowのセッション
        :param observation: 観測
        """
        action_p = self.p_out.eval(session=sess,
                                   feed_dict={self.s: observation})
        return np.random.choice(action_p[0].size, 1,
                                p=action_p[0])[0]

    def estimate_value(self, sess, observation):
        """
        価値関数に基いて価値を推定
        :param sess:        tensorflowのセッション
        :param observation: 観測
        """
        return self.v_out.eval(session=sess,
                               feed_dict={self.s: observation})[0][0]
