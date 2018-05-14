import tensorflow as tf
import numpy as np

from keras.models import Model
from keras.layers import Dense, Input


class NormalModel(object):
    def __init__(self, num_actions, num_states):
        """
        通常の環境用のモデル．
        :param num_actions: 行動数
        :param num_states:  状態の次元数
        """
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

        # 政策と価値関数の出力
        self.model = Model(inputs=inputs,
                           outputs=[action_probs, state_value])
        self.p_out, self.v_out = self.model(self.s)

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
