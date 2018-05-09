from envs.env_wrappers import make_atari, wrap_deepmind

global_step = 0


class Worker(object):
    def __init__(self, model, sess, env_id, thread_id, seed, tmax, batch_size):
        self.thread_id = thread_id
        self.global_step = global_step
        self.tmax = tmax
        self.batch_size = batch_size

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

        # 環境リセット
        obs = self.env.reset()

        # メインループ
        local_step = 0
        while global_step < self.tmax:
            start_step = local_step
            done = False
            while not done and local_step - start_step < self.batch_size:
                # 前の状態を保存
                prev_obs = obs.copy()
                # 行動選択
                action_p = self.model.p_out.eval(
                    session=self.sess,
                    feed_dict={self.model.s: [obs]}
                )
                action = self.env.action_space.sample()
                # 行動を実行し，報酬と次の画面とdoneを観測
                obs, reward, done, _ = self.env.step(action)
                # ステップを進める
                global_step += 1
                local_step += 1
            if done:
                self.env.reset()

            print(self.thread_id, global_step, local_step,
                  local_step - start_step)
