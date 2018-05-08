import argparse

from a3c.trainer import Trainer


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
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--tmax', type=int, default=2000000,
                        help='Number of action selections to finish learning.')
    parser.add_argument('--worker_num', type=int, default=16,
                        help='How many training processes to use')

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
