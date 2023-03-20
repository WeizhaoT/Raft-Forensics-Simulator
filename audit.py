from __future__ import annotations

import time
import os
import sys
import json
import hashlib
import ecdsa
import argparse
import flatten_dict

from colorama import Fore
from tqdm import tqdm
from collections import defaultdict
from os.path import join, isdir
from typing import List, Tuple, Dict, Callable
from file_read_backwards import FileReadBackwards

import node

MIN_PYTHON = (3, 9)
if sys.version_info < MIN_PYTHON:
    sys.exit("Python %s.%s or later is required.\n" % MIN_PYTHON)


HASH = hashlib.sha256()
SECKEY = ecdsa.SigningKey.generate(curve=ecdsa.NIST256p, entropy=ecdsa.util.PRNG(b'srf'))
PUBKEY = SECKEY.get_verifying_key()
MSG = b'rpg'
HASH.update(MSG)
DIGEST = HASH.digest()
SIG = SECKEY.sign(MSG, entropy=ecdsa.util.PRNG(b'sgg'), hashfunc=hashlib.sha256)


FUNC_TIMER = defaultdict(float)


def get_edge_block(path, last=False):
    fp = FileReadBackwards(path) if last else open(path, 'r')

    while True:
        line = fp.readline()
        if line.isspace():
            continue
        if len(line) > 0:
            fp.close()
            return node.Block.fromjson(line)
        else:
            raise EOFError


def binary_search(left, right, left_predicate: Callable[[int], bool]):
    if left > right:
        raise ValueError(f'left {left} larger than right {right}')
    else:
        while left + 1 < right:
            m = (left + right) // 2
            if left_predicate(m):
                right = m
            else:
                left = m

        return left, right


class NodeInfo:
    def __init__(self, id_: int, cc: dict, lc: dict, max_file: int, chunk: Callable[[int], str]) -> None:
        self.id = id_
        self.cc = cc
        self.lc = lc
        self.max_file = max_file
        self.chunk = chunk

        self.last_block = None
        while self.max_file >= 0:
            try:
                self.last_block = get_edge_block(chunk(self.max_file), last=True)
                break
            except EOFError:
                os.remove(chunk(self.max_file))
                self.max_file -= 1

        if self.last_block is None:
            raise ValueError(f'Cannot find any block for node {id_}')
        if self.last_block.h != self.cc['h']:
            raise ValueError(f'Last on-chain block {self.last_block} / CC height {cc["h"]} mismatch for node {id_}')

        self.fp = None
        self.chunk_last_blocks = {self.max_file: self.last_block}
        self.chunk_first_blocks = {0: node.Block(0, 0, -1)}

    def __getitem__(self, k):
        if not isinstance(k, int) or not 0 <= k <= self.max_file:
            raise ValueError(f'No chunk {k} exists for node {self.id} (max {self.max_file})')
        return self.chunk(k)

    @property
    def length(self) -> int:
        return self.cc['h']

    @property
    def final_term(self) -> int:
        return self.cc['t']

    @property
    def cc_voter(self) -> List[int]:
        return self.cc['voters']

    def lc_voter(self, term: int) -> List[int]:
        return self.lc[term]['voters']

    def leader_of(self, term: int):
        return self.lc[term]['leader']

    def first_block_in_chunk(self, k: int):
        if k in self.chunk_first_blocks:
            return self.chunk_first_blocks[k]
        else:
            self.chunk_first_blocks[k] = get_edge_block(self[k], last=False)
            return self.chunk_first_blocks[k]

    def last_block_in_chunk(self, k: int):
        if k in self.chunk_last_blocks:
            return self.chunk_last_blocks[k]
        else:
            self.chunk_last_blocks[k] = get_edge_block(self[k], last=True)
            return self.chunk_last_blocks[k]

    def find_file_range_by_term(self, term: int) -> Tuple[int, int]:
        if term <= 0 or term > self.last_block.t:
            raise ValueError(f'Target term {term} out of range {self.last_block.t}')

        if self.max_file == 0:
            return 0, 0

        first_r = self.first_block_in_chunk(self.max_file)
        if first_r.t < term:
            return self.max_file, self.max_file

        last_l = self.last_block_in_chunk(0)
        if last_l.t > term:
            return 0, 0

        # Find k1, first file chunk that includes first block in "term"
        if term == 1:
            k1 = 0
        else:
            l_, r_ = binary_search(0, self.max_file,
                                   lambda x: self.first_block_in_chunk(x).t >= term)
            last_l = self.last_block_in_chunk(l_)
            k1 = r_ if last_l.t < term else l_

        # Find k2, first file chunk that includes first block with t > "term"
        if term == self.last_block.t:
            k2 = self.max_file
        else:
            _, k2 = binary_search(0, self.max_file,
                                  lambda x: self.last_block_in_chunk(x).t > term)

        assert k1 <= k2
        return k1, k2

    def open(self, k: int):
        self.fp = open(self[k], 'r')

    def close(self):
        if self.fp:
            self.fp.close()

    def read_block(self):
        try:
            return node.Block.fromjson(node.readline_non_empty(self.fp))
        except EOFError:
            return None


def is_valid_int(k, n=0):
    if n > 0:
        return isinstance(k, int) and 0 <= k < n
    else:
        return isinstance(k, int) and 0 <= k


def check_sig(pubkey: ecdsa.VerifyingKey = PUBKEY, sig: bytes = SIG, msg: bytes = MSG, skip: bool = True):
    start = time.time()
    if skip:
        res = True
    else:
        hash = hashlib.sha256()
        hash.update(msg)
        res = pubkey.verify_digest(sig, hash.digest())
    FUNC_TIMER[('check_sig', 'elapsed')] += time.time() - start
    FUNC_TIMER[('check_sig', 'num_calls')] += 1
    return res


def find_matching_height(path: str, target: node.Block) -> node.Block:
    with FileReadBackwards(path) as fp:
        while True:
            line = fp.readline()
            if line.isspace():
                continue
            if len(line) == 0:
                break

            block = node.Block.fromjson(line)
            if block.h == target.h:
                return block

    raise ValueError(f'Fail to find a block with matching height ({target})')


def check_node_integrity(dir_, name, n, bar: tqdm = None) -> Tuple[bool, int, dict, dict, dict]:
    cc, lc_list = None, {}

    timer = defaultdict(float)

    non_chain_files, max_num = [], -1
    dirs = os.listdir(dir_)
    if bar is not None:
        bar.set_description(f'({name}/{n}) Picking files')
        bar.refresh()

    for file in sorted(dirs):
        if file.startswith('blockchain'):
            prefix, postfix = f'blockchain_{name}_', '.jsonl'
            assert file.startswith(prefix) and file.endswith(postfix)
            num = int(file[len(prefix):-len(postfix)])
            if not is_valid_int(num):
                raise ValueError(f'Illegal blockchain file number {num} in {dir_}')

            max_num += 1
            if num != max_num:
                raise ValueError(f'Unexpected blockchain file number {num} (expecting {max_num}) in {dir_}')
        else:
            non_chain_files.append(file)

    if bar is not None:
        bar.set_description(f'({name}/{n}) Analyzing LC and CC')
        bar.refresh()

    for file in non_chain_files:
        if file.startswith('leader'):
            start = time.time()
            with open(join(dir_, file), 'r') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if line.isspace():
                        continue

                    entry = json.loads(line)
                    term = entry.pop('t')
                    if not isinstance(term, int) or term < 0:
                        raise ValueError(f'Invalid term {term} in dir {dir_}')
                    if term in lc_list:
                        raise ValueError(f'Duplicate term {term} in dir {dir_}')
                    if not is_valid_int(entry["leader"], n):
                        raise ValueError(f'Illegal leader {entry["leader"]} of term {term} in dir {dir_}')
                    if len(set(entry['voters'])) != len(entry['voters']):
                        raise ValueError(f'Duplicate voter in LC {entry["voters"]} of term {term} in dir {dir_}')
                    if len(set(entry['voters'])) < n // 2 + 1:
                        raise ValueError(f'Insufficient voter in LC {entry["voters"]} of term {term} in dir {dir_}')
                    if any(not is_valid_int(k, n) for k in entry['voters']):
                        raise ValueError(f'Illegal voter in LC {entry["voters"]} of term {term} in dir {dir_}')

                    for k in entry['voters']:
                        if not check_sig():
                            raise ValueError(f'Illegal sig by voter {k} in LC of term {term} in dir {dir_}')

                    lc_list[term] = entry

            timer[('load', 'leader')] += time.time() - start
        elif file.startswith('commitment'):
            start = time.time()
            if cc is None:
                with open(join(dir_, file), 'r') as f:
                    cc = json.load(f)
                    if not is_valid_int(cc['t']) or not is_valid_int(cc['h']):
                        raise ValueError(f'Invalid CC term/height in dir {dir_}')
                    if len(set(cc['voters'])) != len(cc['voters']):
                        raise ValueError(f'Duplicate voter in CC {entry["voters"]} of term {term} in dir {dir_}')
                    if len(set(cc['voters'])) < n // 2 + 1:
                        raise ValueError(f'Insufficient voter in CC {entry["voters"]} of term {term} in dir {dir_}')
                    if any(not is_valid_int(k, n) for k in cc['voters']):
                        raise ValueError(f'Illegal voter in CC {cc["voters"]} of term {term} in dir {dir_}')

                    for k in cc['voters']:
                        if not check_sig():
                            raise ValueError(f'Illegal sig by voter {k} in CC of term {term} in dir {dir_}')
            else:
                raise ValueError(f'Duplicate CC in dir {dir_}')

            timer[('load', 'commitment')] += time.time() - start

    predecessor, current = None, None
    if bar is not None:
        bar.total = cc['h'] + 1 if bar.total is None else bar.total + cc['h'] + 1
        bar.set_description(f'({name}/{n}) Verifying blockchain data')
        bar.refresh()

    start = time.time()
    for i in range(max_num + 1):
        with open(join(dir_, f'blockchain_{name}_{node.fmt_int(i, max_num)}.jsonl'), 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line.isspace():
                    continue

                current = node.Block.fromjson(line)

                start2 = time.time()
                if predecessor is None:
                    predicate = current != node.Block(0, 0, -1)
                else:
                    predicate = not current.follows(predecessor, check_hash=True) or \
                        current.t not in lc_list or \
                        not check_sig()
                timer[('validate', 'chain')] += time.time() - start2

                if predicate:
                    chain_total = time.time() - start
                    timer[('load', 'chain')] += chain_total - timer[('validate', 'chain')]
                    if bar is not None:
                        bar.set_description(f'Blockchain corrupt')

                    return False, max_num, cc, lc_list, timer

                predecessor = current
                if bar is not None:
                    bar.update()

    chain_total = time.time() - start
    timer[('load', 'chain')] += chain_total - timer[('validate', 'chain')]
    timer[('count', 'chain')] = cc['h'] + 1
    return True, max_num, cc, lc_list, timer


def pair_consistency_check(ni: NodeInfo, nj: NodeInfo):
    timer = {}
    start = time.time()
    conflict = False

    if ni.max_file != nj.max_file:
        nl, nh = (ni, nj) if ni.max_file < nj.max_file else (nj, ni)
        conflict = nl.last_block != find_matching_height(nh[nl.max_file], nl.last_block)
    else:
        if ni.length == nj.length:
            conflict = ni.last_block != nj.last_block
        else:
            nl, nh = (ni, nj) if ni.length < nj.length else (nj, ni)
            conflict = nl.last_block != find_matching_height(nh[nl.max_file], nl.last_block)

    timer[('validate', 'pair consistency')] = time.time() - start
    return conflict, timer


def pair_find_forking_point(ni: NodeInfo, nj: NodeInfo, return_blocks=False) -> Tuple:
    max_file = min(ni.max_file, nj.max_file)
    if ni.last_block_in_chunk(0) != nj.last_block_in_chunk(0):
        forking_file = 0
    elif ni.first_block_in_chunk(max_file) == nj.first_block_in_chunk(max_file):
        forking_file = max_file
    elif max_file > 0 and ni.last_block_in_chunk(max_file - 1) == nj.last_block_in_chunk(max_file - 1):
        if return_blocks:
            bi, bj = ni.first_block_in_chunk(max_file), nj.first_block_in_chunk(max_file)
            return bi.h, (bi, bj)
        else:
            return ni.first_block_in_chunk(max_file).h
    else:
        _, forking_file = binary_search(0, max_file, lambda x: ni.last_block_in_chunk(x) != nj.last_block_in_chunk(x))

    ni.open(forking_file)
    nj.open(forking_file)

    while True:
        bi, bj = ni.read_block(), nj.read_block()
        assert bi is not None and bj is not None
        if bi != bj:
            return (bi.h,) + (((bi, bj),) if return_blocks else ())


def pair_find_adversary(ni: NodeInfo, nj: NodeInfo, return_evidence=False) -> Tuple:
    timer = {}
    start = time.time()
    if ni.final_term == nj.final_term:
        if not return_evidence:
            timer[('find', )] = time.time() - start
            return ([ni.leader_of(ni.final_term)], timer)
        else:
            nl, nh = (ni, nj) if ni.length < nj.length else (nj, ni)
            bh = find_matching_height(nh[nl.max_file], nl.last_block)
            timer[('find', )] = time.time() - start
            return ([ni.leader_of(ni.final_term)], timer, (nl.last_block, bh))

    nl, nh = (ni, nj) if ni.final_term < nj.final_term else (nj, ni)
    term = nl.final_term

    kl, _ = nl.find_file_range_by_term(term)
    kh1, kh2 = nh.find_file_range_by_term(term)

    next_term, kh, first_h, first_l = -1, -1, None, None

    # Find first block of HiNode w/ term >= "term"
    for k in range(kh1, kh2+1):
        nh.open(k)
        while True:
            bh = nh.read_block()
            if bh is None:
                nh.close()
                break

            if bh.t > term:
                next_term = bh.t
                nh.close()
                break
            elif bh.t == term:  # Do not close file if first.term == "term"
                first_h, kh = bh, k
                break

        if next_term >= 0 or first_h is not None:
            break

    # Case 1: Hi has term
    if first_h is not None:
        # Case 1.1: first term block of Lo clearly doesn't match that of Hi
        if kl != kh or first_h.h > nl.last_block.h:
            nh.close()
            if not return_evidence:
                timer[('find', )] = time.time() - start
                return ([nh.leader_of(term)], timer)

            nl.open(kl)
            while True:
                bl = nl.read_block()
                assert bl
                if bl.t == term:
                    nl.close()
                    timer[('find', )] = time.time() - start
                    return ([nh.leader_of(term)], timer, (bl, first_h))

        # Case 1.2: first term block of Lo is in same file as Hi
        nl.open(kh)
        height_matched = False
        while True:
            bl = nl.read_block()
            # After Lo/Hi start reading simultaneously
            if height_matched:  # Impossible if Hi finished & Lo not finished
                if bl is None:
                    nl.close()
                    if kh < nl.max_file:  # Lo not finished
                        nh.close()
                        kh += 1
                        nl.open(kh)
                        nh.open(kh)
                        continue
                    else:  # Lo Finished
                        break

                bh = nh.read_block()
                assert bh is not None
                assert bh.h == bl.h
                assert bh.t >= term
                if bh.t > bl.t:
                    next_term = bh.t
                    nl.close()
                    nh.close()
                    break
                elif bh != bl:
                    nl.close()
                    nh.close()
                    timer[('find', )] = time.time() - start
                    return ([nh.leader_of(term)], timer, (bl, bh))
            # Lo starts catching up with Hi
            elif bl.h == first_h.h:
                if bl != first_h:
                    nl.close()
                    nh.close()
                    timer[('find', )] = time.time() - start
                    return ([nh.leader_of(term)], timer, (bl, first_h))
                height_matched = True
            # Else: Lo.term < "term", continue reading

        if next_term == -1:
            while True:
                bh = nh.read_block()
                if bh is None:
                    nh.close()
                    kh += 1
                    nh.open(kh)
                    continue
                if bh.t > term:
                    nh.close()
                    next_term = bh.t
                    break
    else:
        # Case 2: Hi does not have term
        pass

    assert next_term != -1
    adv = list(set(nh.lc_voter(next_term)).intersection(nl.cc_voter))
    assert len(adv) > 0
    timer[('find', )] = time.time() - start
    return (adv, timer) + (((nl.cc, {'t': next_term} | nh.lc[next_term]),) if return_evidence else ())


def audit_raft_logs(path: str, n: int):
    def chain_file_func(i):
        return lambda c: join(path, f'{node.fmt_int(i, n-1)}', f'blockchain_{node.fmt_int(i, n-1)}_{node.fmt_int(c, maxc[i])}.jsonl')

    meta_timer = defaultdict(float)
    maxc = {}

    checked, adversarial = set([]), set([])

    nodeinfos: List[NodeInfo] = [None] * n
    bar = tqdm()
    for i in range(n):
        dir_ = join(path, node.fmt_int(i, n-1))
        good, nc, cc, lc, timer = check_node_integrity(dir_, node.fmt_int(i, n-1), n, bar)
        for key in timer:
            meta_timer[key] += timer[key]

        maxc[i] = nc

        if good:
            checked.add(i)
            nodeinfos[i] = NodeInfo(i, cc, lc, nc, chain_file_func(i))
        else:
            adversarial.add(i)

    bar.close()
    if adversarial:
        print(f'{Fore.LIGHTRED_EX}Warning: nodes {adversarial} failed integrity checks{Fore.RESET}')

    for i in checked:
        for j in checked:
            if i >= j:
                continue
            conflict, timer = pair_consistency_check(nodeinfos[i], nodeinfos[j])

            for key in timer:
                meta_timer[key] += timer[key]

            if conflict:
                h, (bi, bj) = pair_find_forking_point(nodeinfos[i], nodeinfos[j], return_blocks=True)
                adv, timer, evidence = pair_find_adversary(nodeinfos[i], nodeinfos[j], return_evidence=True)
                # print(i, j, bi, bj, *[e for e in evidence])

                adversarial.update(adv)
                for key in timer:
                    meta_timer[key] += timer[key]

    meta_timer.update(FUNC_TIMER)
    meta_timer = flatten_dict.unflatten(dict(meta_timer))
    meta_timer['adv'] = list(adversarial)
    meta_timer['evidence'] = evidence
    with open(join(path, 'audit.json'), 'w') as f:
        json.dump(meta_timer, f, indent=4)

    print(adversarial)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='path to data', type=str, required=True)
    args = parser.parse_args()

    subdirs = [d for d in os.listdir(args.path) if isdir(join(args.path, d))]
    node_set = defaultdict(int)
    for d in subdirs:
        try:
            assert isinstance(d, str)
            nid = int(d)
            node_set[nid] += 1
        except ValueError:
            continue

    n = max(node_set.keys()) + 1
    for i in range(n):
        if i not in node_set:
            raise ValueError(f'Datadir {args.path} does not include all nodes (expecting {i} from {n})')
        elif node_set[i] != 1:
            raise ValueError(f'Datadir {args.path} has duplicates ({i} has {node_set[i]} replicas)')

    audit_raft_logs(args.path, n)
