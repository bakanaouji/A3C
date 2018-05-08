import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Input


class A3CLSTM(object):
    def __init__(self, num_actions, agent_history_length, frame_width,
                 frame_height):
        # 入力
        self.s = tf.placeholder(
            tf.float32, [None, agent_history_length, frame_width, frame_height]

        )
        inputs = Input(shape=(agent_history_length, frame_width,
                              frame_height))
        # 共通の中間層
        shared = Conv2D(
            16, (8, 8), strides=(4, 4), activation='relu',
            input_shape=(
                agent_history_length,
                frame_width,
                frame_height
            ),
            data_format='channels_first'
        )(inputs)
        shared = Conv2D(32, (4, 4), strides=(2, 2), activation='relu',
                        data_format='channels_first')(shared)
        shared = Flatten()(shared)
        shared = Dense(256, activation='relu')(shared)

        # 政策
        action_probs = Dense(num_actions, activation='softmax')(shared)
        self.policy_network = Model(inputs=inputs, outputs=action_probs)
        self.p_out = self.policy_network(self.s)

        # 価値関数
        state_value = Dense(1)(shared)
        self.value_network = Model(inputs=inputs, outputs=state_value)
        self.v_out = self.value_network(self.s)

