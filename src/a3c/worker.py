from envs.env_wrappers import make_atari, wrap_deepmind


class Worker(object):
    def __init__(self, thread_id, global_step):
        self.thread_id = thread_id
        self.global_step = global_step

    def train(self, env_id, seed):
        # 環境初期化
        env = make_atari(env_id)
        env.seed(seed + self.thread_id)
        env = wrap_deepmind(env)

        # メインループ
        while self.global_step.value >= 0:
            self.global_step.value -= 1
            print(self.thread_id, self.global_step.value)
