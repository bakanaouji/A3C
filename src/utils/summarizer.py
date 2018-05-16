import os
import tensorflow as tf


class Summarizer(object):
    def __init__(self, summary_path):
        sess = tf.keras.backend.get_session()
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        self.summary_writer = tf.summary.FileWriter(summary_path, sess.graph)

    def write(self, values, step):
        summary = tf.Summary()
        for key in values:
            summary.value.add(tag='Info/{}'.format(key),
                              simple_value=values[key])
        self.summary_writer.add_summary(summary, step)
