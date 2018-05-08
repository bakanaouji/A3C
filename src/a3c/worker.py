from envs.env_wrappers import make_atari, wrap_deepmind


class Worker(object):
    def __init__(self, thread_id):
        self.thread_id = thread_id

    def train(self, env_id, seed):
        env = make_atari(env_id)
        env.seed(seed + self.thread_id)
        env = wrap_deepmind(env)
