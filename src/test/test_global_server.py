import unittest
import tensorflow as tf

from a3c.global_server import GlobalServer


class TestGlobalServer(unittest.TestCase):
    def test_global_server(self):
        num_actions = 4
        agent_history_length = 4
        frame_width = 84
        frame_height = 84

        global_server = GlobalServer(num_actions, agent_history_length,
                                     frame_width, frame_height)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        print(global_server.model.model.trainable_weights)


if __name__ == '__main__':
    unittest.main()
