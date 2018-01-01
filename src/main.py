import argparse


def main():
    parser = argparse.ArgumentParser(description="A3C")
    # 環境側のパラメータ
    parser.add_argument('--env_name', default='PongNoFrameskip-v4', help='Environment name')
    parser.add_argument('--width', type=int, default=84, help='Width of resized frame')
    parser.add_argument('--height', type=int, default=84, help='Height of resized frame')
    parser.add_argument('--workers', type=int, default=16, help='How many training processes to use')
