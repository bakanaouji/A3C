import argparse

import gym

from a3c.trainer import Trainer
from models.normal_model import NormalModel


def main():
    parser = argparse.ArgumentParser(description="A3C")
    # 環境側のパラメータ
    parser.add_argument('--env_name', default='CartPole-v0',
                        help='Environment name')
    parser.add_argument('--width', type=int, default=84,
                        help='Width of resized frame')
    parser.add_argument('--height', type=int, default=84,
                        help='Height of resized frame')

    # A3Cのアルゴリズムのパラメータ
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--tmax', type=int, default=2000000,
                        help='Number of action selections to finish learning.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of training cases over which each SGD update is computed.')
    parser.add_argument('--worker_num', type=int, default=8,
                        help='How many training processes to use')
    parser.add_argument('--history_len', type=int, default=1,
                        help='Number of most recent frames experienced '
                             'by the agent that are given as input to the Q-Network.')
    parser.add_argument('--discount_fact', type=float, default=0.99,
                        help='Discount factor gamma used in the A3C update.')

    args = parser.parse_args()

    envs = [gym.make('CartPole-v0') for _ in range(args.worker_num)]

    models = [NormalModel(envs[0].action_space.n,
                          envs[0].observation_space.shape[0])
              for _ in range(args.worker_num + 1)]

    trainer = Trainer(args, envs, models)
    trainer.train()


if __name__ == '__main__':
    main()
