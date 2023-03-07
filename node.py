import json
import os
import numpy as np
from os.path import join
from typing import List


DIR = 'logs'

F_NO = 0
F_YES = 2
F_KEEP = 1


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


class Node:
    def __init__(self, id_, n, adversarial=False) -> None:
        self.id = id_
        self.n = n
        self.neg_sessid = -1
        self.adversarial = adversarial

        self.term = 0

        self.height = 0
        self.blocks = [Block(0, 0, -1, '')]
        self.pending_blocks = {}
        if id_ >= 0:
            filename = f'blockchain_{id_}a.jsonl' if self.adversarial else f'blockchain_{id_}.jsonl'
            self.chain_f = open(join(DIR, filename), 'w')
            self.chain_f.write(f'{self.blocks[0].to_json()}\n')

    @property
    def quorum(self):
        return int(np.ceil((self.n+1)/2))

    @property
    def head(self):
        return self.blocks[0]

    @property
    def tail(self):
        return self.blocks[-1]

    def __lt__(self, node):
        return self.id < node.id

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
            return F_NO

        i_comm = self.head.h - blocks[0].h
        if not i_comm < len(blocks):
            raise ValueError(f'Head {self.head.h} is greater than {blocks[-1].h}')

        if 0 <= i_comm and self.head != blocks[i_comm]:
            return F_NO

        i_ow_input, i_ow_output = -1, -1
        for i, block in enumerate(blocks):
            # check chain
            if i > 0 and not block.follows(blocks[i-1]):
                return F_NO
            i_local = i - i_comm
            # skip when breaking point already set, or no corresponding local position
            if i_local < 0 or i_local > len(self.blocks) or i_ow_input >= 0:
                continue
            #
            if (i_local >= len(self.blocks) or self.blocks[i_local] != block):
                if i_local > 0 and not block.follows(self.blocks[i_local-1]):
                    return F_NO
                i_ow_input, i_ow_output = i, i_local

        if i_comm < -len(self.blocks):
            return F_KEEP

        if i_ow_input >= 0:
            self.blocks = self.blocks[:i_ow_output] + blocks[i_ow_input:]

        return F_YES

    def read_blocks_since(self, h_start):
        if h_start >= self.head.h:
            return self.blocks[h_start-self.head.h:]

        files = [join(DIR, f) for f in os.listdir(DIR) if os.path.isfile(
            join(DIR, f)) and f.startswith(f'blockchain_{self.id}') and f.endswith('.jsonl')]

        blocks = {}
        for filename in files:
            with open(filename, 'r') as f:
                for line in f.readlines():
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

    def commit(self, t, h) -> bool:
        i = h - self.head.h
        if i <= 0 or i >= len(self.blocks) or self.blocks[i].t != t:
            return False

        for block in self.blocks[1:i+1]:
            self.chain_f.write(f'{block.to_json()}\n')

        self.chain_f.flush()
        self.blocks = self.blocks[i:]
        return True
