import numpy as np
import argparse

import node
from event import TestEvents
from network import Network
from delays import BaseDelay, ConstantDelay, UniformDelay, ModuloDelay, ModuloSplitDelay, ModuloRandomDelay


def main(period, maxtime):
    n = 5
    # BAD VOTE:
    delay = ModuloSplitDelay(n, 100, 300)
    events, plans = TestEvents.TEST_BAD_VOTE_1(tx_count=100, tx_interval=50, tx_retry=20)

    # NORMAL:
    # delay = ModuloRandomDelay(n, 0, step=100, rand_mult=2.0)
    # events, plans = TestEvents.TEST_LEADER_CHANGE_2(tx_count=111, tx_interval=50, tx_retry=20)
    net = Network(n, delay, tx_retry_time=500, datadir='logs', adversary=1, debug=True)
    net.run(period, maxtime, events, plans, sleep=.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--period', type=int, default=50, help='period')
    parser.add_argument('-M', '--maxtime', type=int, default=10000, help='max time')
    parser.add_argument('-s', '--seed', type=int, default=120, help='Random Seed')
    parser.add_argument('-b', '--blockchain-filesize', type=int, default=0, help='blockchain filesize')
    parser.add_argument('-l', '--log-filesize', type=int, default=0, help='log filesize')
    parser.add_argument('-u', '--uncommitted-filesize', type=int, default=0, help='uncommitted filesize')

    args = parser.parse_args()

    new_fs = {
        'blockchain': args.blockchain_filesize,
        'log': args.log_filesize,
        'uncommitted': args.uncommitted_filesize
    }

    for key in list(new_fs.keys()):
        if new_fs[key] == 0:
            del new_fs[key]

    node.FILESIZES.update(new_fs)

    np.random.seed(args.seed)

    main(args.period, args.maxtime)
