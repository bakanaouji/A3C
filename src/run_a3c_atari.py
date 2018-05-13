import argparse

from a3c.trainer import Trainer
from envs.env_wrappers import make_atari, wrap_deepmind
from models.atari_model import AtariModel


def main():
    parser = argparse.ArgumentParser(description="A3C")
    # 環境側のパラメータ
    parser.add_argument('--env_name', default='BreakoutNoFrameskip-v4',
                        help='Environment name')
    parser.add_argument('--width', type=int, default=84,
                        help='Width of resized frame')
    parser.add_argument('--height', type=int, default=84,
                        help='Height of resized frame')

    # A3Cのアルゴリズムのパラメータ
    parser.add_argument('--seed', help='random seed', type=int, default=0)
    parser.add_argument('--tmax', type=int, default=50e6,
                        help='Number of action selections to finish learning.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of training cases over which each SGD update is computed.')
    parser.add_argument('--worker_num', type=int, default=8,
                        help='How many training processes to use')
    parser.add_argument('--learn_rate', type=float, default=7e-4,
                        help='Learning rate used by RMSProp.')
    parser.add_argument('--discount_fact', type=float, default=0.99,
                        help='Discount factor gamma used in the A3C update.')
    parser.add_argument('--decay', type=float, default=0.99,
                        help='RMSProp decay factor.')
    parser.add_argument('--entropy_weight', type=float, default=0.01,
                        help='Weight of entropy regularization.')

    args = parser.parse_args()

    envs = [wrap_deepmind(make_atari(args.env_name)) for _ in
            range(args.worker_num)]

    models = [AtariModel(envs[0].action_space.n, args.width, args.height)
              for _ in range(args.worker_num + 1)]

    trainer = Trainer(args, envs, models)
    trainer.train()


if __name__ == '__main__':
    main()
