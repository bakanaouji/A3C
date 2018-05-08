import unittest

from keras.utils.vis_utils import plot_model
from models.a3c_lstm import A3CLSTM


class TestModel(unittest.TestCase):
    def test_a3c_lstm(self):
        num_actions = 4
        agent_history_length = 4
        frame_width = 84
        frame_height = 84
        a3c_lstm = A3CLSTM(num_actions, agent_history_length, frame_width,
                           frame_height)
        s_shape = a3c_lstm.s.get_shape().as_list()
        self.assertEqual(s_shape, [None, agent_history_length, frame_width,
                                   frame_height])
        q_values_shape = a3c_lstm.p_out.get_shape().as_list()
        self.assertEqual(q_values_shape, [None, num_actions])
        plot_model(a3c_lstm.policy_network, show_shapes=True,
                   show_layer_names=True, to_file='policy_network.png')
        plot_model(a3c_lstm.value_network, show_shapes=True,
                   show_layer_names=True, to_file='value_network.png')


if __name__ == '__main__':
    unittest.main()
