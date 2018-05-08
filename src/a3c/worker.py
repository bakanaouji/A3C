from envs.env_wrappers import make_atari, wrap_deepmind

global_step = 0


class Worker(object):
    def __init__(self, env_id, thread_id, seed, tmax):
        self.thread_id = thread_id
        self.global_step = global_step
        self.tmax = tmax

        # 環境初期化
        self.env = make_atari(env_id)
        self.env.seed(seed + thread_id)
        self.env = wrap_deepmind(self.env)

    def train(self):
        global global_step

        # 環境リセット
        obs = self.env.reset()

        # メインループ
        local_step = 0
        while global_step < self.tmax:
            # 前の状態を保存
            prev_obs = obs.copy()
            # 行動選択
            action = 1
            # 行動を実行し，報酬と次の画面とdoneを観測
            obs, reward, done, _ = self.env.step(action)
            if done:
                self.env.reset()

            global_step += 1
            local_step += 1
            print(self.thread_id, global_step, local_step)
