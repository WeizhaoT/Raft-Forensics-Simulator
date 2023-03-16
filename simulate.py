import numpy as np
import argparse

import time
import node
from event import TestEvents
from network import Network
from delays import BaseDelay, ConstantDelay, UniformDelay, ModuloDelay, ModuloSplitDelay, ModuloRandomDelay


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--period', type=int, default=50, help='period')
    parser.add_argument('-M', '--maxtime', type=int, default=-1, help='max time')
    parser.add_argument('-s', '--seed', type=int, default=time.time_ns() % 2**32, help='Random Seed')
    parser.add_argument('-b', '--blockchain-filesize', type=int, default=0, help='blockchain filesize')
    parser.add_argument('-l', '--log-filesize', type=int, default=0, help='log filesize')
    parser.add_argument('-u', '--uncommitted-filesize', type=int, default=0, help='uncommitted filesize')
    parser.add_argument('-x', '--transactions', type=int, default=100, help='transaction count')
    parser.add_argument('-i', '--tx-interval', type=int, default=50, help='transaction interval')
    parser.add_argument('-e', '--election-freq', type=int, default=20, help='election interval')
    parser.add_argument('-w', '--wait', type=float, default=0., help='waiting time')
    parser.add_argument('--depth', type=float, default=0.5, help='attack depth')
    parser.add_argument('-d', '--debug', action='store_true', help='debug')
    parser.add_argument('--fork', action='store_true')
    parser.add_argument('--bvote', action='store_true')

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

    n = 5
    # BAD VOTE:
    if args.bvote:
        prefix = 'badvote'
        delay = ModuloSplitDelay(n, 100, 1000)
        # events, plans = TestEvents.TEST_BAD_VOTE_1(tx_count=100, tx_interval=50, tx_retry=20)
        events, plans = TestEvents.BAD_VOTE(tx_count=args.transactions,
                                            tx_interval=args.tx_interval, tx_retry=20, le_freq=args.election_freq, depth=args.depth)
        datadir = f'{prefix}-{args.transactions}-{args.election_freq}-{args.blockchain_filesize}-{args.depth:4.2f}'
    elif args.fork:
        prefix = 'fork'
        delay = ModuloRandomDelay(n, 0, step=100, rand_mult=2.0)
        events, plans = TestEvents.LEADER_FORK(tx_count=args.transactions,
                                               tx_interval=args.tx_interval, tx_retry=20, le_freq=args.election_freq, depth=args.depth)
        datadir = f'{prefix}-{args.transactions}-{args.election_freq}-{args.blockchain_filesize}-{args.depth:4.2f}'
    else:
        prefix = 'normal'
        delay = ModuloRandomDelay(n, 0, step=100, rand_mult=2.0)
        events, plans = TestEvents.NORMAL(tx_count=args.transactions,
                                          tx_interval=args.tx_interval, tx_retry=20, le_freq=args.election_freq)
        datadir = f'{prefix}-{args.transactions}-{args.election_freq}-{args.blockchain_filesize}'

    # FORK:
    # delay = ModuloRandomDelay(n, 0, step=100, rand_mult=2.0)
    # events, plans = TestEvents.TEST_FORK_1(tx_count=args.transactions, tx_interval=args.tx_interval, tx_retry=20)

    # NORMAL:
    if args.maxtime < 0:
        args.maxtime = int(max(max([e.t for e in events]), args.transactions * args.tx_interval) * 1.1) + 10000

    # datadir = 'logs'
    net = Network(n, delay, tx_retry_time=10 * args.tx_interval, datadir=datadir, adversary=4, debug=args.debug)
    net.run(args.period, args.maxtime, events, plans, sleep=args.wait)
