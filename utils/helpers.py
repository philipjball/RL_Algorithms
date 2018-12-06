import argparse


def check_train_test(value):
    if value in ['train', 'test']:
        return value
    else:
        raise argparse.ArgumentTypeError("%s is not 'train' or 'test'" % value)


def check_env(value):
    if value in ['FlappyBird-v0', 'CartPole-v0', 'CartPole-v1']:
        return value
    else:
        raise argparse.ArgumentTypeError("%s is not a gym environment" % value)