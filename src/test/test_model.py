import unittest
import tensorflow as tf

from keras.utils.vis_utils import plot_model

from models.atari_model import AtariModel
from models.normal_model import NormalModel


class TestModel(unittest.TestCase):
    def test_atari_model(self):
        num_actions = 4
        frame_width = 84
        frame_height = 84

        model = AtariModel(num_actions, frame_width, frame_height)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        s_shape = model.s.get_shape().as_list()
        self.assertEqual(s_shape, [None, 4, frame_width, frame_height])
        p_out_shape = model.p_out.get_shape().as_list()
        self.assertEqual(p_out_shape, [None, num_actions])
        v_out_shape = model.v_out.get_shape().as_list()
        self.assertEqual(v_out_shape, [None, 1])
        plot_model(model.model, show_shapes=True,
                   show_layer_names=True, to_file='atari_model.png')

    def test_normal_model(self):
        num_actions = 2
        model = NormalModel(num_actions, 4)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        s_shape = model.s.get_shape().as_list()
        self.assertEqual(s_shape, [None, 4])
        p_out_shape = model.p_out.get_shape().as_list()
        self.assertEqual(p_out_shape, [None, num_actions])
        v_out_shape = model.v_out.get_shape().as_list()
        self.assertEqual(v_out_shape, [None, 1])
        plot_model(model.model, show_shapes=True,
                   show_layer_names=True, to_file='normal_model.png')


if __name__ == '__main__':
    unittest.main()
