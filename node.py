from __future__ import annotations
import json
import os
import io
import shutil
import numpy as np
from os.path import join
from typing import List, Dict
from copy import deepcopy
from enum import Enum
from file_read_backwards import FileReadBackwards


DIR = 'logs'


def readline_non_empty(f: io.TextIOWrapper) -> str:
    for _ in range(10000):
        line: str = f.readline()
        if len(line) == 0:
            raise EOFError()
        if not line.isspace():
            return line

    raise NotImplementedError('readline_non_empty encountered too many empty lines')


class Rtype(Enum):
    REJ = -1
    NO = 0
    KEEP = 1
    YES = 2


class Block:
    def __init__(self, t, h, pt, tx=''):
        self.t = t
        self.h = h
        self.pt = pt
        self.tx = tx

    @staticmethod
    def fromjson(jsonstr: str):
        obj = json.loads(jsonstr)
        return Block(obj['t'], obj['h'], obj['pt'], obj['tx'])

    def __str__(self):
        return f'{self.t}-{self.h}({self.pt})'

    def to_json(self):
        return f'{{\"t\": {self.t}, \"h\": {self.h}, \"pt\": {self.pt}, \"tx\": \"{self.tx}\"}}'

    def __lt__(self, block):
        if self.t != block.t:
            return self.t < block.t
        else:
            return self.h < block.h

    def __gt__(self, block):
        if self.t != block.t:
            return self.t > block.t
        else:
            return self.h > block.h

    def __eq__(self, block):
        return self.t == block.t and self.h == block.h and self.pt == block.pt and self.tx == block.tx

    def __add__(self, h: int):
        return Block(self.t, self.h + h, -1, -1, '')

    def follows(self, block):
        return self.pt == block.t

    @staticmethod
    def strlist(blocks: List[Block]):
        if len(blocks) == 0:
            return '[]'

        block_strs, term, last_h = [], -1, -1
        for j, block in enumerate(blocks + [Block(blocks[-1].t+1, blocks[-1].h+1, 0)]):
            if term < block.t:
                # block_strs.append(f'"{term}, {j}, {block}"')
                if j > 0:
                    if block.h == last_h + 2:
                        block_strs.append(f'"{str(blocks[j-1])}"')
                    elif block.h >= last_h + 3:
                        block_strs.extend(['"..."', f'"{str(blocks[j-1])}"'])

                if j < len(blocks):
                    term = block.t
                    last_h = block.h
                    block_strs.append(f'"{str(block)}"')

        return '[' + ', '.join(block_strs) + ']'


class Node:
    def __init__(self, id_, n, adversarial=False) -> None:
        self.id: int = id_
        self.n: int = n
        self.adversarial: bool = adversarial

        self.term: int = 0

        self.blocks: List[Block] = [Block(0, 0, -1, '')]

        self.cc: List[int] = []
        self.lc: Dict[int, int] = {}

        self.listeners: set(Node) = set([])

        self.asked = -1
        self.acked = -1

        if id_ >= 0:
            self.log_f: io.TextIOWrapper = open(join(DIR, self.filename('log')), 'w')
            self.chain_f: io.TextIOWrapper = open(join(DIR, self.filename('blockchain')), 'w')
            self.leader_f: io.TextIOWrapper = open(join(DIR, self.filename('leader')), 'w')

            self.chain_f.write(f'{self.blocks[0].to_json()}\n')
        else:
            self.chain_f = None
            self.leader_f = None

    @property
    def quorum(self):
        return int(np.ceil((self.n+1)/2))

    @property
    def head(self):
        return self.blocks[0]

    @property
    def tail(self):
        return self.blocks[-1]

    @property
    def freshness(self):
        return self.blocks[-1].t, self.blocks[-1].h

    @property
    def peers(self):
        return [self] + list(self.listeners)

    @property
    def name(self):
        return f'a{self.id}' if self.adversarial else f'{self.id}'

    def filename(self, prefix):
        return f'{prefix}_{self.name}.jsonl'

    def write_log(self, t, log):
        self.log_f.write(f'[ {t:6d} ms ] {str(log)}\n')
        self.log_f.flush()

    def validate_cc(self, block: Block, cc: List[int]) -> bool:
        return True

    def validate_lc(self, term: int, lc: List[int]) -> bool:
        return True

    def __lt__(self, node: Node):
        return self.id < node.id

    def set_asked(self) -> bool:
        ret = self.asked < self.head.h
        self.asked = self.head.h
        return ret

    def set_acked(self) -> bool:
        ret = self.acked < self.tail.h
        self.acked = self.tail.h
        return ret

    def reset_leader_prog(self):
        self.asked = -1
        self.acked = -1

    def clear_listeners(self):
        self.listeners = set([])

    def add_listeners(self, nodes: List[Node]):
        self.listeners.update(n for n in nodes if n.id != self.id)

    def remove_listener(self, node: Node):
        self.listeners.remove(node)

    def accept_leader(self, term: int, fresh_term: int, fresh_height: int, leader_id: int, lc: List[int]):
        assert term not in self.lc
        self.lc[term] = {'l': leader_id, 'ft': fresh_term, 'fh': fresh_height, 'lc': lc}
        json.dump({'t': term, 'ft': fresh_term, 'fh': fresh_height, 'leader': leader_id, 'voters': lc}, self.leader_f)
        self.leader_f.write('\n')
        self.leader_f.flush()

    def increment_term(self) -> int:
        self.term += 1
        return self.term

    def update_term(self, term):
        if term >= self.term:
            self.term = term

    def accept_tx(self, tx: str) -> Block:
        # self.height += 1
        new_block = Block(self.term, self.tail.h+1, self.tail.t, tx)
        self.blocks.append(new_block)
        return new_block

    def handle_append_entries(self, blocks: List[Block]) -> int:
        assert len(blocks) > 0
        if not self.head < blocks[-1]:
            return Rtype.NO

        i_comm = self.head.h - blocks[0].h
        if not i_comm < len(blocks):
            raise ValueError(f'Head {self.head.h} is greater than {blocks[-1].h}')

        if 0 <= i_comm and self.head != blocks[i_comm]:
            return Rtype.REJ

        i_ow_input, i_ow_output = -1, -1
        for i, block in enumerate(blocks):
            # check chain
            if i > 0 and not block.follows(blocks[i-1]):
                return Rtype.NO

            i_local = i - i_comm
            # skip when breaking point already set, or no corresponding local position
            if i_local < 0 or i_local > len(self.blocks) or i_ow_input >= 0:
                continue
            #
            if (i_local >= len(self.blocks) or self.blocks[i_local] != block):
                if i_local > 0 and not block.follows(self.blocks[i_local-1]):
                    return Rtype.REJ
                i_ow_input, i_ow_output = i, i_local

        if i_comm < -len(self.blocks):
            return Rtype.KEEP

        if i_ow_input >= 0:
            self.blocks = self.blocks[:i_ow_output] + blocks[i_ow_input:]

        return Rtype.YES

    def read_blocks_since(self, h_start):
        if h_start >= self.head.h:
            return self.blocks[h_start-self.head.h:]

        files = [join(DIR, f) for f in os.listdir(DIR) if os.path.isfile(
            join(DIR, f)) and f.startswith(f'blockchain_{self.name}') and f.endswith('.jsonl')]

        files = sorted(files)
        blocks = {}
        for filename in files:
            with open(filename, 'r') as f:
                for line in f.readlines():
                    if not line.isspace():
                        block = Block.fromjson(line)
                        if block.h > h_start:
                            blocks[block.h] = block

        i, block_list = h_start+1, []
        while i in blocks:
            block_list.append(blocks[i])
            del blocks[i]
            i += 1

        if len(blocks) > 0:
            raise ValueError(f'Corrupt block data: block {i} missing')

        return block_list + self.blocks[1:]

    def has(self, block: Block) -> bool:
        if self.tail.t < block.t or self.tail.h < block.h:
            return False
        if (self.head.t < block.t and self.head.h >= block.h) or (self.head.t > block.t and self.head.h < block.h):
            return False

        if self.head == block:
            return True
        if self.head < block:
            for b in self.blocks:
                if b == block:
                    return True
            return False

        files = [join(DIR, f) for f in os.listdir(DIR) if os.path.isfile(
            join(DIR, f)) and f.startswith(f'blockchain_{self.name}') and f.endswith('.jsonl')]

        for filename in sorted(files, reverse=True):
            try:
                start_dist, end_dist = -1, -1
                with open(filename, 'r') as f:
                    start = Block.fromjson(readline_non_empty(f))
                    if start.h > block.h or start.h < 0:
                        if start.t < block.t:
                            return False
                        start_dist = block.h - start.h
                        continue

                with FileReadBackwards(filename) as f:
                    end = Block.fromjson(readline_non_empty(f))
                    if end.h < block.h:
                        return False
                    end_dist = end.h - block.h
                    if end_dist < start_dist:
                        temp = end
                        while temp.h > block.h and temp.h >= 0:
                            temp = Block.fromjson(readline_non_empty(f))

                        return block == temp

                with open(filename, 'r') as f:
                    try:
                        while True:
                            temp = Block.fromjson(readline_non_empty(f))
                            if temp.h < 0 or temp.h >= block.h:
                                break
                        return block == temp
                    except EOFError:
                        return False
            except EOFError:
                raise

    def commit(self, t, h, cc) -> bool:
        i = h - self.head.h
        if i <= 0 or i >= len(self.blocks) or self.blocks[i].t != t:
            return False

        if not self.validate_cc(self.blocks[i], cc):
            return False

        self.cc = cc
        with open(join(DIR, self.filename('commitment')), 'w') as f:
            json.dump({'t': t, 'h': h, 'voters': cc}, f)

        for block in self.blocks[1:i+1]:
            self.chain_f.write(f'{block.to_json()}\n')

        self.chain_f.flush()
        self.blocks = self.blocks[i:]
        return True

    def steal_from(self, node: Node):
        self.blocks = deepcopy(node.blocks)
        self.cc = deepcopy(node.cc)
        self.lc = deepcopy(node.lc)

        self.chain_f.close()
        self.leader_f.close()

        PREFIXES = ['blockchain', 'commitment', 'leader']
        for file in os.listdir(DIR):
            if any(file.startswith(f'{p}_{self.name}') for p in PREFIXES):
                os.remove(join(DIR, file))

        for file in os.listdir(DIR):
            for p in PREFIXES:
                fullpf = f'{p}_{node.name}'
                if file.startswith(fullpf):
                    postfix = file[len(fullpf):]
                    shutil.copyfile(join(DIR, file), join(DIR, f'{p}_{self.name}{postfix}'))

        # print([str(b) for b in self.read_blocks_since(0)])

        self.chain_f = open(join(DIR, self.filename('blockchain')), 'a')
        self.leader_f = open(join(DIR, self.filename('leader')), 'a')

    def flush_uncommitted(self):
        with open(join(DIR, self.filename('uncommitted')), 'w') as f:
            for block in self.blocks[1:]:
                f.write(block.to_json() + '\n')
