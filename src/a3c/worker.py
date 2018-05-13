import numpy as np
import tensorflow as tf

global_step = 0


class Worker(object):
    def __init__(self, sess, global_server, env, model, thread_id, args):
        self.global_server = global_server
        self.thread_id = thread_id
        self.global_step = global_step
        self.tmax = args.tmax
        self.batch_size = args.batch_size
        self.discount_fact = args.discount_fact
        self.entropy_weight = args.entropy_weight

        # initialize environment
        self.env = env
        self.num_actions = self.env.action_space.n

        # initialize session
        self.sess = sess

        # initialize model
        self.model = model
        self.weights = self.model.model.trainable_weights

        # initialize update operation
        self.A, self.R, self.apply_grads = self.build_training_op()

        # initialize synchronize operation
        self.sync_parameter = self.build_sync_op()

    def build_training_op(self):
        A = tf.placeholder(tf.int32, [None])
        R = tf.placeholder(tf.float32, [None])

        log_prob = tf.log(tf.reduce_sum(self.model.p_out *
                                        tf.one_hot(A, depth=self.num_actions),
                                        axis=1, keepdims=True))
        p_loss = tf.reduce_mean(-log_prob *
                                tf.stop_gradient(R - self.model.v_out))
        v_loss = tf.reduce_mean(tf.square(R - self.model.v_out))
        entropy = tf.reduce_mean(tf.reduce_sum(self.model.p_out *
                                               tf.log(self.model.p_out),
                                               axis=1, keepdims=True))
        loss = p_loss - entropy * self.entropy_weight + v_loss * 0.5

        grads = tf.gradients(loss, self.weights)

        apply_grads = self.global_server.optimizer.apply_gradients(
            zip(grads, self.global_server.weights))

        return A, R, apply_grads

    def build_sync_op(self):
        weights = self.model.weights
        src_weights = self.global_server.weights
        sync_parameter = [weight.assign(src_weight) for weight, src_weight
                          in zip(weights, src_weights)]
        return sync_parameter

    def train(self):
        global global_step

        # reset environment
        obs = self.env.reset()

        # メインループ
        local_step = 0
        total_reward = 0
        episode_num = 0
        episode_len = 0
        while global_step < self.tmax:
            self.sess.run(self.sync_parameter)

            s_batch = []
            a_batch = []
            r_history = []

            start_step = local_step
            done = False
            lr = None

            # run episode
            while not done and local_step - start_step < self.batch_size:
                # choose action
                action = self.model.take_action(self.sess, [obs])
                lr = self.global_server.scheduler.value()

                # perform action
                next_obs, reward, done, _ = self.env.step(action)
                total_reward += reward

                # append observation, reward and action to batch
                s_batch.append(obs)
                a_batch.append(action)
                r_history.append(reward)

                obs = next_obs

                # advance step
                global_step += 1
                local_step += 1
                episode_len += 1
            a_batch = np.int32(np.array(a_batch))
            s_batch = np.float32(np.array(s_batch))

            R = 0
            if done:
                obs = self.env.reset()
            else:
                # bootstrap from last state
                R = self.model.estimate_value(self.sess, [obs])

            batch_size = len(s_batch)

            # make reward batch
            r_batch = np.zeros(batch_size)
            for i in reversed(range(batch_size)):
                R = r_history[i] + self.discount_fact * R
                r_batch[i] = R

            # update global shared parameter
            self.sess.run(self.apply_grads,
                          feed_dict={self.A: a_batch,
                                     self.R: r_batch,
                                     self.model.s: s_batch,
                                     self.global_server.lr: lr
                                     }
                          )
            if done:
                print('thread: {0}, '
                      'global step: {1}, '
                      'local step: {2}, '
                      'total reward: {3} '
                      'episode len: {4} '
                      'learning rate: {5}'
                      .format(self.thread_id, global_step, local_step,
                              total_reward, episode_len, lr))
                total_reward = 0
                episode_len = 0
                episode_num += 1
