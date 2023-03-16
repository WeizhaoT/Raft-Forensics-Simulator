from __future__ import annotations
import json
import os
import io
import shutil
import hashlib
import numpy as np
from os.path import join
from typing import List, Dict, Callable, Set
from copy import deepcopy
from enum import Enum
from file_read_backwards import FileReadBackwards


CACHESIZE = 600

FILESIZES = {
    'blockchain': 10000,
    'leader': -1,
    'log': 1000,
    'uncommitted': -1
}


def get_edge_block(path, last=False):
    fp = FileReadBackwards(path) if last else open(path, 'r')

    while True:
        line = fp.readline()
        if line.isspace():
            continue
        if len(line) > 0:
            fp.close()
            return Block.fromjson(line)
        else:
            raise EOFError


def fmt_int(i, imax, zero=True):
    fmt = f'{{:0{len(str(imax))}d}}' if zero else f'{{:{len(str(imax))}d}}'
    return fmt.format(i)


def ms_to_str(t: int | float):
    s, ms = divmod(int(t), 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f'{h:02d}:{m:02d}:{s:02d}.{ms:03d}'


def readline_non_empty(f: io.TextIOWrapper) -> str:
    for _ in range(10000):
        line: str = f.readline()
        if len(line) == 0:
            raise EOFError()
        if not line.isspace():
            return line

    raise NotImplementedError('readline_non_empty encountered too many empty lines')


class Rtype(Enum):
    REJ3 = -3
    REJ2 = -2
    REJ1 = -1
    NO = 0
    KEEP = 1
    YES = 2


class Block:
    INT_BYTES = 8

    def __init__(self, t, h, pt, tx=''):
        self.t = t
        self.h = h
        self.pt = pt
        self.tx = tx

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

    def to_bytearray(self):
        return int.to_bytes(self.t, Block.INT_BYTES, 'big') + \
            int.to_bytes(self.h, Block.INT_BYTES, 'big') + \
            int.to_bytes(self.pt, Block.INT_BYTES, 'big', signed=True) + \
            bytearray(self.tx, 'utf-8')

    def follows(self, block, check_hash=False):
        if check_hash:
            hash = hashlib.sha256()
            hash.update(block.to_bytearray())
            digest = bytearray(hash.digest())
            digest2 = digest.copy()
            return digest == digest2 and self.pt == block.t

        return self.pt == block.t

    @staticmethod
    def fromjson(jsonstr: str):
        obj = json.loads(jsonstr)
        return Block(obj['t'], obj['h'], obj['pt'], obj['tx'])

    def frombytearray(b: bytearray):
        t = int.from_bytes(b[:Block.INT_BYTES])
        h = int.from_bytes(b[Block.INT_BYTES:2*Block.INT_BYTES])
        pt = int.from_bytes(b[2*Block.INT_BYTES:3*Block.INT_BYTES], signed=True)
        tx = b[3*Block.INT_BYTES:].decode()
        return Block(t, h, pt, tx)

    @staticmethod
    def strlist(blocks: List[Block], full=False):
        if len(blocks) == 0:
            return '[]'
        if full:
            return '[' + ', '.join(f'"{str(block)}"' for block in blocks) + ']'

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
    def __init__(self, id_, n, adversarial=False, dir_='logs') -> None:
        self.id: int = id_
        self.n: int = n
        self.adversarial: bool = adversarial

        self.term: int = 0

        self.cache: List[Block] = []
        self.blocks: List[Block] = [Block(0, 0, -1, '')]

        self.cc: List[int] = []

        self.listeners: Set[Node] = set([])

        self.asked = -1
        self.acked = -1

        self.linequota: Dict[str, int] = deepcopy(FILESIZES)
        self.filecount: Dict[str, int] = {p: 0 for p in FILESIZES}
        self.fds: Dict[str, io.TextIOWrapper] = {}

        self.dir = dir_

        if id_ >= 0:
            for p, fs in FILESIZES.items():
                if fs <= 0:
                    self.fds[p] = open(join(self.dir, self.filename(p)), 'w')
                else:
                    self.fds[p] = open(join(self.dir, self.filename(p, 0)), 'w')

            self.write_items('blockchain', self.blocks, lambda x: x.to_json())

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

    def write_items(self, prefix: str, items, str_call: Callable = lambda x: x):
        assert prefix in FILESIZES
        if not isinstance(items, list):
            return self.write_items(prefix, [items], str_call)
        elif len(items) == 0:
            return

        written = 0
        while written < len(items):
            writable = len(items) - written
            if FILESIZES[prefix] > 0:
                writable = min(self.linequota[prefix], writable)

            for i in range(written, written + writable):
                self.fds[prefix].write(str_call(items[i]) + '\n')

            self.fds[prefix].flush()
            written += writable
            if FILESIZES[prefix] <= 0:
                break

            self.linequota[prefix] -= writable
            if self.linequota[prefix] <= 0:
                self.increment_filecount(prefix)
                self.linequota[prefix] = FILESIZES[prefix]

    def write_log(self, t: int, log: str):
        self.write_items('log', f'[{ms_to_str(t)}] {str(log)}')

    def increment_filecount(self, prefix):
        self.filecount[prefix] += 1
        fc = self.filecount[prefix]
        self.fds[prefix].close()
        self.fds[prefix] = open(join(self.dir, self.filename(prefix, fc)), 'w')

        if len(str(fc)) == len(str(fc-1)):
            return

        for file in os.listdir(self.dir):
            starter, ender = f'{prefix}_{self.name}', '.jsonl'
            if not file.startswith(starter) or not file.endswith(ender):
                continue

            try:
                c = int(file[len(starter):-len(ender)].strip('_'))
            except ValueError:
                print(f'Warning: cannot read count from file {file}')
                continue

            if c != fc:
                os.rename(join(self.dir, file), join(self.dir, self.filename(prefix, c)))

    def filename(self, prefix, fc=None):
        if fc is None:
            return f'{prefix}_{self.name}.jsonl'
        else:
            nd = len(str(self.filecount[prefix]))
            fc = f'{{:0{nd}d}}'.format(fc)
            return f'{prefix}_{self.name}_{fc}.jsonl'

    def validate_cc(self, block: Block, cc: List[int]) -> bool:
        return True

    def validate_lc(self, term: int, lc: List[int]) -> bool:
        return True

    def __lt__(self, node: Node):
        return self.id < node.id

    def set_asked(self, h) -> bool:
        ret = self.asked < h
        self.asked = max(self.asked, h)
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
        assert term > self.term or leader_id == self.id
        self.write_items('leader', json.dumps({'t': term, 'ft': fresh_term,
                         'fh': fresh_height, 'leader': leader_id, 'voters': lc}))

    def increment_term(self) -> int:
        self.term += 1
        return self.term

    def update_term(self, term):
        if term >= self.term:
            self.term = term

    def accept_tx(self, tx: str) -> Block:
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
            return Rtype.REJ1

        i_ow_input, i_ow_output = -1, -1
        for i, block in enumerate(blocks):
            # check chain
            if i > 0 and not block.follows(blocks[i-1]):
                return Rtype.REJ3

            i_local = i - i_comm
            # skip when breaking point already set, or no corresponding local position
            if i_local < 0 or i_local > len(self.blocks) or i_ow_input >= 0:
                continue
            #
            if (i_local >= len(self.blocks) or self.blocks[i_local] != block):
                if i_local > 0 and not block.follows(self.blocks[i_local-1]):
                    return Rtype.REJ2
                i_ow_input, i_ow_output = i, i_local

        if i_comm < -len(self.blocks):
            return Rtype.KEEP

        if i_ow_input >= 0:
            self.blocks = self.blocks[:i_ow_output] + blocks[i_ow_input:]

        return Rtype.YES

    def read_blocks_since(self, h_start):
        if h_start >= self.head.h:
            return self.blocks[h_start-self.head.h:]
        elif self.cache and h_start >= self.cache[0].h:
            return self.cache[h_start-self.cache[0].h:] + self.blocks

        files = [join(self.dir, f) for f in os.listdir(self.dir) if os.path.isfile(
            join(self.dir, f)) and f.startswith(f'blockchain_{self.name}') and f.endswith('.jsonl')]

        files = sorted(files)
        blocks, last_h = [], h_start
        for filename in files:
            last = get_edge_block(filename, last=True)
            if last.h < h_start:
                continue

            with open(filename, 'r') as f:
                for line in f.readlines():
                    if not line.isspace():
                        block = Block.fromjson(line)
                        if block.h == last_h + 1:
                            blocks.append(block)
                            last_h += 1

        if not blocks or blocks[-1].h != self.head.h:
            raise ValueError(f'Corrupt block data: block {block[-1].h + 1} missing')

        return blocks + self.blocks[1:]

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

        files = [join(self.dir, f) for f in os.listdir(self.dir) if os.path.isfile(
            join(self.dir, f)) and f.startswith(f'blockchain_{self.name}') and f.endswith('.jsonl')]

        for filename in sorted(files, reverse=True):
            start_dist, end_dist = -1, -1
            with open(filename, 'r') as f:
                try:
                    start = Block.fromjson(readline_non_empty(f))
                except EOFError:
                    continue

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
                    try:
                        while temp.h > block.h and temp.h >= 0:
                            temp = Block.fromjson(readline_non_empty(f))
                    except EOFError:
                        raise EOFError(
                            f'Narrowed block {block} in file {filename} ({start} -- {end}), but block not found')

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

        return False

    def commit(self, t, h, cc) -> Rtype:
        i = h - self.head.h
        if i >= len(self.blocks):
            return Rtype.KEEP

        if i <= 0:
            return Rtype.NO

        if self.blocks[i].t != t or not self.validate_cc(self.blocks[i], cc):
            return Rtype.REJ1

        self.cc = cc

        with open(join(self.dir, self.filename('commitment')), 'w') as f:
            json.dump({'t': t, 'h': h, 'voters': cc}, f)

        self.write_items('blockchain', self.blocks[1:i+1], lambda b: b.to_json())
        if i >= CACHESIZE:
            self.cache = self.blocks[i-CACHESIZE:i]
        else:
            self.cache = self.cache[max(0, len(self.cache) + i - CACHESIZE):] + self.blocks[:i]

        self.blocks = self.blocks[i:]
        return Rtype.YES

    def steal_from(self, node: Node):
        self.cache = deepcopy(node.cache)
        self.blocks = deepcopy(node.blocks)
        self.cc = deepcopy(node.cc)
        self.filecount = deepcopy(node.filecount)
        self.linequota = deepcopy(node.linequota)

        prefixes = ['blockchain', 'commitment', 'leader']

        for p in prefixes:
            if p in FILESIZES and self.fds[p] is not None:
                self.fds[p].close()

        for file in os.listdir(self.dir):
            if any(file.startswith(f'{p}_{self.name}') for p in prefixes):
                os.remove(join(self.dir, file))

        for file in os.listdir(node.dir):
            for p in prefixes:
                fullpf = f'{p}_{node.name}'
                if file.startswith(fullpf):
                    postfix = file[len(fullpf):]
                    shutil.copyfile(join(node.dir, file), join(self.dir, f'{p}_{self.name}{postfix}'))

        for p in prefixes:
            if p not in FILESIZES:
                continue
            if FILESIZES[p] <= 0:
                self.fds[p] = open(join(self.dir, self.filename(p)), 'a')
            else:
                self.fds[p] = open(join(self.dir, self.filename(p, self.filecount[p])), 'a')

    def flush_uncommitted(self):
        self.write_items('uncommitted', self.blocks[1:], lambda b: b.to_json())
