import unittest

import cv2
import gym
import numpy as np

from envs.env_wrappers import NoOpResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, \
    FireResetEnv, WarpFrame, FrameStack, ClipRewardEnv, ScaledFloatFrame


class TestEnvWrappers(unittest.TestCase):
    def test_noop_reset_env(self):
        """
        no_op_maxフレーム分だけ何もしてなければおｋ
        """
        env = gym.make("BreakoutNoFrameskip-v4")
        env = NoOpResetEnv(env, no_op_max=30)
        env.reset()

    def test_max_and_skip_env(self):
        """
        action_repeatフレーム分だけ行動を繰り返している，
        かつ，
        状態が前フレームの観測値との最大値であればおｋ
        """
        env = gym.make("BreakoutNoFrameskip-v4")
        env = MaxAndSkipEnv(env, skip=4)
        env.reset()
        env.step(0)

    def test_episodic_life_env(self):
        """
        Breakoutなどのライフがあるゲームで，
        1ライフ失う毎にエピソードが終わったことになっていればok
        """
        env = gym.make("BreakoutNoFrameskip-v4")
        env = EpisodicLifeEnv(env)
        env.reset()
        while True:
            _, _, done, _ = env.step(1)
            if done:
                env.reset()
                break
            env.render()

    def test_fire_reset_env(self):
        """
        ゲームが開始されていれば観測値が前フレームと変わっているはずなので，
        観測値が前フレームと比べて変化していればok
        """
        env = gym.make("BreakoutNoFrameskip-v4")
        env = FireResetEnv(env)
        pre_observation = env.reset()
        observation, _, _, _ = env.step(0)
        self.assertFalse((observation == pre_observation).all())

    def test_warp_frame(self):
        """
        cv2で画像を保存してみて，グレースケールならok
        """
        env = gym.make("BreakoutNoFrameskip-v4")
        env = WarpFrame(env)
        observation = env.reset()
        cv2.imwrite("test_warp_frame.png", observation[:, :, 0] * 255)

    def test_frame_stack(self):
        """
        スタックするフレーム数分だけ次元数が増加していればok
        """
        k = 4
        env = gym.make("BreakoutNoFrameskip-v4")
        expected_observation_shape = (
            env.observation_space.shape[0], env.observation_space.shape[1],
            env.observation_space.shape[2] * k)
        env = FrameStack(env, k)
        observation = env.reset()
        self.assertEqual(expected_observation_shape,
                         np.array(observation).shape)
        self.assertEqual(env.observation_space.shape,
                         np.array(observation).shape)
        observation, _, _, _ = env.step(0)
        self.assertEqual(env.observation_space.shape,
                         np.array(observation).shape)

    def test_clipped_rewards_wrapper(self):
        """
        報酬が-1~1の間に入っていればok
        """
        env = gym.make("BreakoutNoFrameskip-v4")
        env = ClipRewardEnv(env)
        env.reset()
        _, reward, _, _ = env.step(0)
        self.assertLessEqual(reward, 1.0)
        self.assertGreaterEqual(reward, -1.0)

    def test_scaled_float_frame(self):
        """
        状態の各値が0~1に入っていればok
        """
        env = gym.make("BreakoutNoFrameskip-v4")
        env = ScaledFloatFrame(env)
        observation = env.reset()
        self.assertTrue((observation <= 1.0).all())
        self.assertTrue((observation >= 0.0).all())
