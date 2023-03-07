from enum import Enum
from typing import List

from node import Node, Block


class Etype(Enum):
    LEAD = -1
    ACK = 0
    CMT = 1
    TX = 2
    REP = 3
    R2 = 4

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Event:
    def __init__(self, subject: Node, type: Etype, t, *args) -> None:
        self.subject = subject
        self.t = t
        self.type = type
        self.args = args

    def __lt__(self, obj):
        return (self.t, self.type, self.subject.id, self.args) < (obj.t, obj.type, obj.subject.id, obj.args)

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
