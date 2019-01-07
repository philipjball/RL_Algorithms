import argparse
import os
from utils.helpers import *


def main(args):
    if 'CartPole-v' in args.env:
        if args.testfile:
            testfile = args.testfile
        else:
            testfile = './params/trained_params_gym_cp2.pth'
        command_str = 'python runExp.py --env ' + args.env + ' --frame_stack 1 --testfile ' + testfile\
                      + ' --num_episodes ' + str(args.num_episodes) + ' --frame_skip 1'
    elif args.env == 'FlappyBird-v0':
        if args.testfile:
            testfile = args.testfile
        else:
            testfile = './params/trained_params_gym_fb.pth'
        command_str = 'python runExp.py --testfile ' + testfile + ' --num_episodes ' + str(args.num_episodes)
    elif args.env == 'VizdoomBasic-v0':
        if args.testfile:
            testfile = args.testfile
        else:
            testfile = './params/trained_params_gym_vizdoombasic.pth'
        command_str = 'python runExp.py --testfile ' + testfile + ' --env ' + args.env + ' --num_episodes '\
                      + str(args.num_episodes)

    if args.visualise:
        command_str +=  ' --visualise True'
    os.system(command_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise the agents that I\'ve trained.')
    parser.add_argument('--env', type=check_env, default='FlappyBird-v0',
                        help='set to required environment')
    parser.add_argument('--testfile', type=str, default=None,
                        help='path for own model')
    parser.add_argument('--num_episodes', type=int, default=20,
                        help='test over this many episodes')
    parser.add_argument('--visualise', type=bool, default=False,
                        help='visualise the agent')
    arguments = parser.parse_args()
    main(arguments)