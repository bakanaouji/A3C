import numpy as np

from envs.env_wrappers import make_atari, wrap_deepmind

global_step = 0


class Worker(object):
    def __init__(self, model, sess, env_id, thread_id, seed, tmax, batch_size,
                 discount_fact):
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
        self.model = model

    def train(self):
        global global_step

        # reset environment
        obs = self.env.reset()

        # メインループ
        local_step = 0
        while global_step < self.tmax:
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