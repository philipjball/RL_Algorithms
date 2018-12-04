import argparse
import os


def check_env(value):
    if value in ['FlappyBird-v0', 'CartPole-v0']:
        return value
    else:
        raise argparse.ArgumentTypeError("%s is not a gym environment" % value)


def main(args):
    if args.env == 'CartPole-v0':
        os.system('python runExp.py --env CartPole-v0 --mode train --reward_shaping False '
                  '--frame_stack 1 --frame_skip 2 --final_exp_frame 5000 --save_freq 5000 '
                  '--reset_target 500 --memory_size 10000')
    elif args.env == 'FlappyBird-v0':
        os.system('python runExp.py --mode train')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the same training setup for a given environment.')
    parser.add_argument('--env', type=check_env, default='FlappyBird-v0',
                        help='set to required environment')
    arguments = parser.parse_args()
    main(arguments)