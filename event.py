from __future__ import annotations
from enum import Enum
from typing import List

from node import Node, Block


class Etype(Enum):
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
    def __init__(self, subject: Node, type: Etype, t, *args) -> None:
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
            if isinstance(arg, Node):
                argstrs.append(f'"Node-{arg.id}"')
            elif isinstance(arg, list):
                block_strs, term, last_h = [], -1, -1
                if len(arg) == 0:
                    argstrs.append('[]')
                    continue

                if isinstance(arg[0], int):
                    argstrs.append('[' + ', '.join(str(i) for i in arg) + ']')
                else:
                    for j, block in enumerate(arg + [Block(arg[-1].t+1, arg[-1].h+1, 0)]):
                        if term < block.t:
                            # block_strs.append(f'"{term}, {j}, {block}"')
                            if j > 0:
                                if block.h == last_h + 2:
                                    block_strs.append(f'"{str(arg[j-1])}"')
                                elif block.h >= last_h + 3:
                                    block_strs.extend(['"..."', f'"{str(arg[j-1])}"'])

                            if j < len(arg):
                                term = block.t
                                last_h = block.h
                                block_strs.append(f'"{str(block)}"')

                    argstrs.append('[' + ', '.join(block_strs) + ']')
            elif isinstance(arg, (int, float)):
                argstrs.append(f'{str(arg)}')
            else:
                argstrs.append(f'"{str(arg)}"')
        argstr = ', '.join(argstrs)
        return f'{{"time": {int(self.t)}, "type": "{self.type.name}", "node": {int(self.subject.id)}, "args": [{argstr}]}}'


class TestEvents:
    @staticmethod
    def TEST_LEADER_CHANGE_1(tx_count, tx_interval, tx_retry):
        return [Event(None, Etype.LEAD, 0, 0)] + \
            [Event(None, Etype.TX, tc * tx_interval, f'{tc:4d}', tx_retry) for tc in range(1, tx_count+1)] +\
            [Event(None, Etype.LEAD, 1000, -1), Event(None, Etype.LEAD, 1100, 2)]

    @staticmethod
    def TEST_LEADER_CHANGE_2(tx_count, tx_interval, tx_retry):
        return [Event(None, Etype.LEAD, 0, 0)] + \
            [Event(None, Etype.TX, tc * tx_interval, f'{tc:4d}', tx_retry) for tc in range(1, tx_count+1)] +\
            [Event(None, Etype.LEAD, 1000, -1), Event(None, Etype.LEAD, 1100, 2)] + \
            [Event(None, Etype.LEAD, 2000, -1), Event(None, Etype.LEAD, 2100, 4)]
