import numpy as np
import tensorflow as tf

from envs.env_wrappers import make_atari, wrap_deepmind
from models.a3c_lstm import A3CLSTM

global_step = 0


class Worker(object):
    def __init__(self, sess, global_model, env_id, thread_id, seed, tmax,
                 batch_size, discount_fact, history_len, width, height):
        self.thread_id = thread_id
        self.global_step = global_step
        self.tmax = tmax
        self.batch_size = batch_size
        self.discount_fact = discount_fact

        # initialize environment
        self.env = make_atari(env_id)
        self.env.seed(seed + thread_id)
        self.env = wrap_deepmind(self.env)

        # initialize session
        self.sess = sess

        # initialize model
        self.model = A3CLSTM(thread_id, self.env.action_space.n, history_len,
                             width, height)
        self.global_model = global_model

    def build_training_op(self):
        A = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])

        log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.model.p_out, labels=A)
        p_loss = tf.reduce_mean(log_prob *
                                tf.stop_gradient(R - self.model.v_out))
        v_loss = tf.reduce_mean(tf.square(R - self.model.v_out))
        entropy = tf.reduce_mean(tf.reduce_sum(self.model.p_out *
                                               tf.log(self.model.p_out),
                                               axis=1,
                                               keep_dims=True))
        loss = p_loss - entropy * 0.01 + v_loss * 0.5

        # define optimizer
        optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-3, decay=0.99,
                                              epsilon=1e-5)

        weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           'model' + self.thread_id)
        grads = tf.gradients(loss, weights_params)

        apply_grads = optimizer.apply_gradients(grads)

    def train(self):
        global global_step

        # reset environment
        obs = self.env.reset()

        # メインループ
        local_step = 0
        while global_step < self.tmax:
            self.model.update_param(self.global_model.policy_network,
                                    self.global_model.value_network)

            s_batch = []
            a_batch = []
            r_history = []

            start_step = local_step
            done = False

            # run episode
            while not done and local_step - start_step < self.batch_size:
                # 前の状態を保存
                prev_obs = obs.copy()

                # choose action
                action = self.model.take_action(self.sess, [obs])

                # perform action
                obs, reward, done, _ = self.env.step(action)

                # append observation, reward and action to batch
                s_batch.append(prev_obs)
                a_batch.append(action)
                r_history.append(reward)

                # advance step
                global_step += 1
                local_step += 1

            R = 0
            if done:
                self.env.reset()
            else:
                # bootstrap from last state
                R = self.model.estimate_value(self.sess, [obs])

            episode_len = len(s_batch)

            # make reward batch
            r_batch = np.zeros(episode_len)
            for i in reversed(range(episode_len)):
                R = r_history[i] + self.discount_fact * R
                r_batch[i] = R

            print(self.thread_id, global_step, local_step, episode_len)
