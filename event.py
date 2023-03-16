from __future__ import annotations
from enum import Enum
from typing import List, Callable

import node


class Etype(Enum):
    INFO = -10
    SET_FORK = -6
    UNSET_FORK = -5
    FORK = -4
    BADVOTELEAD = -3
    AUTOLEAD = -2
    LEAD = -1
    ACK = 0
    REJ = 1
    CMT = 11
    TX = 12
    REP = 13
    R2 = 14

    def __lt__(self, other: Enum):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Event:
    def __init__(self, subject: node.Node, type: Etype, t, *args) -> None:
        self.subject = subject
        self.t = t
        self.type = type
        self.args = args

    @property
    def subid(self):
        return self.subject.id if self.subject else -1

    def __lt__(self, obj: Event):
        return (self.t, self.type, self.subid, self.args) < (obj.t, obj.type, obj.subid, obj.args)

    def __str__(self):
        argstrs = []
        for arg in self.args:
            if isinstance(arg, node.Node):
                argstrs.append(f'"Node-{arg.id}"')
            elif isinstance(arg, list):
                if len(arg) == 0:
                    argstrs.append('[]')
                    continue

                if isinstance(arg[0], (int, str, float)):
                    argstrs.append('[' + ', '.join(str(i) for i in arg) + ']')
                elif isinstance(arg[0], node.Block):
                    argstrs.append(node.Block.strlist(arg))
                else:
                    raise NotImplementedError
            elif isinstance(arg, (int, float)):
                argstrs.append(f'{str(arg)}')
            else:
                argstrs.append(f'"{str(arg)}"')
        argstr = ', '.join(argstrs)
        return f'{{"time": {int(self.t)}, "type": "{self.type.name}", "node": {int(self.subject.id)}, "args": [{argstr}]}}'


class EventPlan:
    def __init__(self, type: Etype, count: int, start: int, interval: int, args_func: Callable[[int], tuple] = lambda x: tuple([])) -> None:
        self.type = type
        self.count = count
        self.start = start
        self.interval = interval
        self.args_func = args_func

        self.maxt = self.start + self.interval * (self.count - 1)
        self.c = 0

    def events_until(self, t):
        events = []
        t0 = self.start + self.c * self.interval
        while t0 <= t and self.c < self.count:
            events.append(Event(None, self.type, t0, *self.args_func(self.c)))
            self.c += 1
            t0 += self.interval

        return events

    @staticmethod
    def TX_PLAN(tx_count, tx_interval, tx_retry):
        return EventPlan(Etype.TX, tx_count, tx_interval, tx_interval, lambda i: (node.fmt_int(i+1, tx_count), tx_retry))

    @staticmethod
    def LE_PLAN(le_count, le_interval):
        return EventPlan(Etype.AUTOLEAD, le_count, le_interval, le_interval, lambda _: tuple())


class TestEvents:
    @staticmethod
    def TEST_TX(tx_count, tx_interval, tx_retry):
        return [Event(None, Etype.LEAD, 0, 0)], [EventPlan.TX_PLAN(tx_count, tx_interval, tx_retry)]

    @staticmethod
    def TEST_LEADER_CHANGE_1(tx_count, tx_interval, tx_retry):
        return [Event(None, Etype.LEAD, 0, 0), Event(None, Etype.AUTOLEAD, 1100, -1)],\
            [EventPlan.TX_PLAN(tx_count, tx_interval, tx_retry)]

    @staticmethod
    def TEST_LEADER_CHANGE_2(tx_count, tx_interval, tx_retry):
        return [Event(None, Etype.LEAD, 0, 0), Event(None, Etype.AUTOLEAD, 1100), Event(None, Etype.AUTOLEAD, 2100)],\
            [EventPlan.TX_PLAN(tx_count, tx_interval, tx_retry)]

    @staticmethod
    def TEST_FORK_1(tx_count, tx_interval, tx_retry):
        return [Event(None, Etype.SET_FORK, 0), Event(None, Etype.LEAD, 0, 0), Event(None, Etype.LEAD, 1000, -1),
                Event(None, Etype.LEAD, 1100, 1)], [EventPlan.TX_PLAN(tx_count, tx_interval, tx_retry)]

    @staticmethod
    def TEST_BAD_VOTE_1(tx_count, tx_interval, tx_retry):
        return [Event(None, Etype.SET_BAD_VOTE, 0), Event(None, Etype.LEAD, 0, 0), Event(None, Etype.AUTOLEAD, 2500)],\
            [EventPlan.TX_PLAN(tx_count, tx_interval, tx_retry)]

    @staticmethod
    def NORMAL(tx_count, tx_interval, tx_retry, le_freq):
        return [Event(None, Etype.LEAD, 0, 0)], \
            [EventPlan.TX_PLAN(tx_count, tx_interval, tx_retry), EventPlan.LE_PLAN(
                int(tx_count // le_freq) if le_freq > 0 else 0, int(tx_interval * le_freq))],

    @staticmethod
    def BAD_VOTE(tx_count, tx_interval, tx_retry, le_freq, depth):
        t = tx_interval + int(depth * (tx_count - 1) * tx_interval)
        return [Event(None, Etype.BADVOTELEAD, t), Event(None, Etype.LEAD, 0, 0)], \
            [EventPlan.TX_PLAN(tx_count, tx_interval, tx_retry), EventPlan.LE_PLAN(
                int(tx_count // le_freq) if le_freq > 0 else 0, int(tx_interval * le_freq))],

    @staticmethod
    def LEADER_FORK(tx_count, tx_interval, tx_retry, le_freq, depth):
        t = tx_interval + int(depth * (tx_count - 1) * tx_interval)
        return [Event(None, Etype.FORK, t), Event(None, Etype.LEAD, 0, 0)], \
            [EventPlan.TX_PLAN(tx_count, tx_interval, tx_retry), EventPlan.LE_PLAN(
                int(tx_count // le_freq) if le_freq > 0 else 0, int(tx_interval * le_freq))],
