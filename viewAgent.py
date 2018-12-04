import argparse
import os


def check_env(value):
    if value in ['FlappyBird-v0', 'CartPole-v0']:
        return value
    else:
        raise argparse.ArgumentTypeError("%s is not a gym environment" % value)


def main(args):
    if args.env == 'CartPole-v0':
        if args.testfile:
            testfile = args.testfile
        else:
            testfile = './params/trained_params_gym_cp.pth'
        os.system('python runExp.py --env CartPole-v0 --frame_stack 1 --visualise True --testfile ' + testfile +
                  ' --num_episodes ' + str(args.num_episodes))
    elif args.env == 'FlappyBird-v0':
        if args.testfile:
            testfile = args.testfile
        else:
            testfile = './params/trained_params_gym_fb.pth'
        os.system('python runExp.py --testfile ' + testfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise the agents that I\'ve trained.')
    parser.add_argument('--env', type=check_env, default='FlappyBird-v0',
                        help='set to required environment')
    parser.add_argument('--testfile', type=str, default=None,
                        help='path for own model')
    parser.add_argument('--num_episodes', type=int, default=20,
                        help='test over this many episodes')
    arguments = parser.parse_args()
    main(arguments)