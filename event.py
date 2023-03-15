from __future__ import annotations
from enum import Enum
from typing import List

import node


class Etype(Enum):
    INFO = -10
    SET_FORK = -6
    SET_BAD_VOTE = -5
    UNSET_FORK = -4
    UNSET_BAD_VOTE = -3
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


class TestEvents:
    @staticmethod
    def TEST_TX(tx_count, tx_interval, tx_retry):
        return [Event(None, Etype.LEAD, 0, 0)] + \
            [Event(None, Etype.TX, tc * tx_interval, f'{tc:4d}', tx_retry) for tc in range(1, tx_count+1)]

    @staticmethod
    def TEST_LEADER_CHANGE_1(tx_count, tx_interval, tx_retry):
        return [Event(None, Etype.LEAD, 0, 0)] + \
            [Event(None, Etype.TX, tc * tx_interval, f'{tc:4d}', tx_retry) for tc in range(1, tx_count+1)] +\
            [Event(None, Etype.AUTOLEAD, 1100, -1)]

    @staticmethod
    def TEST_LEADER_CHANGE_2(tx_count, tx_interval, tx_retry):
        return [Event(None, Etype.LEAD, 0, 0)] + \
            [Event(None, Etype.TX, tc * tx_interval, f'{tc:4d}', tx_retry) for tc in range(1, tx_count+1)] +\
            [Event(None, Etype.AUTOLEAD, 1100)] + \
            [Event(None, Etype.AUTOLEAD, 2100)]

    @staticmethod
    def TEST_FORK_1(tx_count, tx_interval, tx_retry):
        return [Event(None, Etype.SET_FORK, 0), Event(None, Etype.LEAD, 0, 0)] + \
            [Event(None, Etype.TX, tc * tx_interval, f'{tc:4d}', tx_retry) for tc in range(1, tx_count+1)] +\
            [Event(None, Etype.LEAD, 1000, -1), Event(None, Etype.LEAD, 1100, 1)]

    @staticmethod
    def TEST_BAD_VOTE_1(tx_count, tx_interval, tx_retry):
        return [Event(None, Etype.SET_BAD_VOTE, 0), Event(None, Etype.LEAD, 0, 0)] + \
            [Event(None, Etype.TX, tc * tx_interval, f'{tc:4d}', tx_retry) for tc in range(1, tx_count+1)] +\
            [Event(None, Etype.AUTOLEAD, 1100)]
