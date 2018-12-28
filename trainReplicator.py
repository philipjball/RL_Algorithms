import argparse
import os
from utils.helpers import *


def main(args):
    if 'Vizdoom' in args.env:
        os.system('python runExp.py --env ' + args.env + ' --mode train')
    elif 'CartPole-v' in args.env:
        os.system('python runExp.py --env ' + args.env + ' --mode train --reward_shaping False '
                  '--frame_stack 1 --frame_skip 1 --final_exp_frame 10000 --save_freq 5000 '
                  '--reset_target 500 --memory_size 1000000 --gamma 0.95 --batch_size 32 '
                  '--weight_decay 1e-4 --lr_scheduler [5000,10000,15000,20000,25000,30000]')
    elif args.env == 'FlappyBird-v0':
        os.system('python runExp.py --mode train')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the same training setup for a given environment.')
    parser.add_argument('--env', type=check_env, default='FlappyBird-v0',
                        help='set to required environment')
    arguments = parser.parse_args()
    main(arguments)
