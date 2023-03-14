import numpy as np
import argparse

from event import TestEvents
from network import Network
from delays import BaseDelay, ConstantDelay, UniformDelay, ModuloDelay, ModuloRandomDelay


def main(period, maxtime):
    n = 5
    net = Network(n, ModuloRandomDelay(n, 0, step=100, rand_mult=2.0), tx_retry_time=500, adversary=1)
    events = TestEvents.TEST_BAD_VOTE_1(tx_count=100, tx_interval=50, tx_retry=20)
    net.run(period, maxtime, events, sleep=.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--period', type=int, default=50, help='period')
    parser.add_argument('-M', '--maxtime', type=int, default=10000, help='max time')
    parser.add_argument('-s', '--seed', type=int, default=120, help='Random Seed')

    args = parser.parse_args()

    np.random.seed(args.seed)

    main(args.period, args.maxtime)
