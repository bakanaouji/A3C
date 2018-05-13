import argparse

import gym

from a3c.trainer import Trainer
from models.normal_model import NormalModel


def main():
    parser = argparse.ArgumentParser(description="A3C")
    # 環境側のパラメータ
    parser.add_argument('--env_name', default='CartPole-v0',
                        help='Environment name')

    # A3Cのアルゴリズムのパラメータ
    parser.add_argument('--seed', help='random seed', type=int, default=0)
    parser.add_argument('--tmax', type=int, default=1000000,
                        help='Number of action selections to finish learning.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of training cases over which each SGD update is computed.')
    parser.add_argument('--worker_num', type=int, default=8,
                        help='How many training processes to use')
    parser.add_argument('--learn_rate', type=float, default=5e-3,
                        help='Learning rate used by RMSProp.')
    parser.add_argument('--discount_fact', type=float, default=0.99,
                        help='Discount factor gamma used in the A3C update.')
    parser.add_argument('--decay', type=float, default=0.99,
                        help='RMSProp decay factor.')
    parser.add_argument('--entropy_weight', type=float, default=0.01,
                        help='Weight of entropy regularization.')

    args = parser.parse_args()

    envs = [gym.make(args.env_name) for _ in range(args.worker_num)]

    models = [NormalModel(envs[0].action_space.n,
                          envs[0].observation_space.shape[0])
              for _ in range(args.worker_num + 1)]

    trainer = Trainer(args, envs, models)
    trainer.train()


if __name__ == '__main__':
    main()
