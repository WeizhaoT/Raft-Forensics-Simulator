from __future__ import annotations

import time
import os
import json
import hashlib
import ecdsa
import argparse
import flatten_dict

from tqdm import tqdm
from collections import defaultdict
from os.path import join, isdir
from typing import List, Tuple, Set
from file_read_backwards import FileReadBackwards

import node

HASH = hashlib.sha256()
SECKEY = ecdsa.SigningKey.generate(curve=ecdsa.NIST256p, entropy=ecdsa.util.PRNG(b'srf'))
PUBKEY = SECKEY.get_verifying_key()
MSG = b'rpg'
HASH.update(MSG)
DIGEST = HASH.digest()
SIG = SECKEY.sign(MSG, entropy=ecdsa.util.PRNG(b'sgg'), hashfunc=hashlib.sha256)


FUNC_TIMER = defaultdict(float)


def is_valid_int(k, n):
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


def read_all_blocks(path):
    blocks = []
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if line.isspace():
                continue
            if len(line) > 0:
                blocks.append(node.Block.fromjson(line))
            else:
                return blocks


def get_file_size(path):
    size = 0
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if line.isspace():
                continue
            if len(line) > 0:
                size += 1
            else:
                return size


def find_file_range_by_term(term, k1_init, k2_init, f) -> Tuple[int, int]:
    k2 = k2_init
    last = get_edge_block(f(k2), last=True)

    # Find first file k2 of Node-High whose final block has a higher term
    while True:
        if last.t > term:
            if k2 == 0:
                break
            else:
                last = get_edge_block(f(k2-1), last=True)
                if last.t > term:
                    k2 -= 1
                    continue
                else:
                    break
        else:
            k2 += 1
            last = get_edge_block(f(k2), last=True)
            if last.t > term:
                break
            else:
                continue

    k1 = min(k1_init, k2)
    first = get_edge_block(f(k1), last=False)
    # Find first file k1 of Node-High which includes first block with a non-lower term
    while True:
        if first.t < term:
            if k1 == k2:
                break
            else:
                first = get_edge_block(f(k1+1), last=False)
                if first.t < term:
                    k1 += 1
                    continue
                else:
                    break
        else:
            if k1 == 0:
                break
            else:
                k1 -= 1
                first = get_edge_block(f(k1), last=False)
                if first.t < term:
                    break
                else:
                    continue

    last = get_edge_block(f(k1), last=True)
    if last.t < term:
        k1 += 1

    return k1, k2


def check_node_integrity(dir_, name, n) -> Tuple[bool, int, dict, dict, dict]:
    cc, lc_list = None, {}

    timer = defaultdict(float)

    non_chain_files, max_num = [], -1
    for file in sorted(os.listdir(dir_)):
        if file.startswith('blockchain'):
            prefix, postfix = f'blockchain_{name}_', '.jsonl'
            assert file.startswith(prefix) and file.endswith(postfix)
            num = int(file[len(prefix):-len(postfix)])
            if not is_valid_int(num, -1):
                raise ValueError(f'Illegal blockchain file number {num} in {dir_}')

            max_num += 1
            if num != max_num:
                raise ValueError(f'Unexpected blockchain file number {num} (expecting {max_num}) in {dir_}')
        else:
            non_chain_files.append(file)

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
                    if not is_valid_int(cc['t'], -1) or not is_valid_int(cc['h'], -1):
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
    bar = tqdm(desc=f'Verifying blockchain data', total=cc['h'] + 1)

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
                    bar.set_description(f'Blockchain corrupt')
                    bar.close()
                    return False, max_num, cc, lc_list, timer

                predecessor = current
                bar.update()

    chain_total = time.time() - start
    timer[('load', 'chain')] += chain_total - timer[('validate', 'chain')]
    timer[('count', 'chain')] = cc['h'] + 1
    bar.close()
    return True, max_num, cc, lc_list, timer


def pair_consistency_check(i, j, fi, fj, maxc):
    ci, cj = maxc[i], maxc[j]

    timer = {}
    start = time.time()
    consistency = None

    if ci != cj:
        f, c, fo, _ = (fi, ci, fj, cj) if ci < cj else (fj, cj, fi, ci)
        last = None
        while True:
            try:
                last = get_edge_block(f(c), last=True)
                break
            except EOFError:
                os.remove(f(c))
                c -= 1
                if ci < cj:
                    maxc[i] = c
                else:
                    maxc[j] = c

        assert last is not None

        with FileReadBackwards(fo(c)) as fp:
            while True:
                line = fp.readline()
                if line.isspace():
                    continue
                if len(line) == 0:
                    raise ValueError(f'Fail to find a block with matching height ({last.to_json()})')

                block = node.Block.fromjson(line)
                if block.h == last.h:
                    consistency = block == last
                    break
    else:
        lasti, lastj = None, None
        while True:
            try:
                lasti = get_edge_block(fi(ci), last=True)
                break
            except EOFError:
                os.remove(fi(ci))
                ci -= 1
                maxc[i] = ci

        while True:
            try:
                lastj = get_edge_block(fj(cj), last=True)
                break
            except EOFError:
                os.remove(fj(cj))
                cj -= 1
                maxc[j] = cj

        assert lasti is not None and lastj is not None
        if lasti.h == lastj.h:
            consistency = lasti == lastj
        else:
            fo, c, last = (fj, ci, lasti) if lasti.h < lastj.h else (fi, cj, lastj)

            with FileReadBackwards(fo(c)) as fp:
                while True:
                    line = fp.readline()
                    if line.isspace():
                        continue
                    if len(line) == 0:
                        raise ValueError(f'Fail to find a block with matching height ({last.to_json()})')

                    block = node.Block.fromjson(line)
                    if block.h == last.h:
                        consistency = block == last
                        break

    assert consistency is not None
    timer[('validate', 'pair consistency')] = time.time() - start
    return consistency, timer


def pair_find_adversary(size, fi, fj, ci, cj, cci, ccj, lci, lcj):
    timer = {}
    start = time.time()
    if cci['t'] == ccj['t']:
        timer[('find', )] = time.time() - start
        return [lci[cci['t']]['leader']], timer

    fl, fh, cl, ch, ccl, cch, lcl, lch = (fi, fj, ci, cj, cci, ccj, lci, lcj) if cci['t'] < ccj['t'] else \
        (fj, fi, cj, ci, ccj, cci, lcj, lci)

    term, kl = min(cci['t'], ccj['t']), ccl['h'] // size

    # Locate first block of term owned by Node-Low
    while True:
        first = get_edge_block(fl(kl))
        if first.t >= term and kl > 0:
            kl -= 1
            continue
        else:
            break

    if first.t < term:
        last = get_edge_block(fl(kl), last=True)
        if last.t < term:
            kl += 1

    kh1, kh2 = find_file_range_by_term(term, min(kl, ch), min(kl, ch), fh)

    assert kh2 >= kh1
    next_term = -1
    for k in range(kh1, kh2+1):
        blocks_h: List[node.Block] = read_all_blocks(fh(k))
        if kl <= k <= cl:
            blocks_l: List[node.Block] = read_all_blocks(fl(k))
            for bh, bl in zip(blocks_h, blocks_l):
                assert bh.h == bl.h
                if bh.t == bl.t:
                    if bh != bl:
                        timer[('find', )] = time.time() - start
                        return [lch[term]['leader']], timer
                elif bh.t < bl.t:
                    continue
                else:
                    next_term = bh.t
                    break

            if next_term != -1:
                break

            assert len(blocks_h) >= len(blocks_l)
            blocks_h = blocks_h[len(blocks_l) + 1:]

        if len(blocks_h) == 0 or blocks_h[-1].t <= term:
            continue

        if blocks_h[0].t > term:
            next_term = blocks_h[0].t
        else:
            for b in (blocks_h[1:]):
                if b.t > term:
                    next_term = b.t
                    break
        break

    assert next_term != -1
    LC = set(lch[next_term]['voters'])
    CC = set(ccl['voters'])
    adv = list(CC.intersection(LC))
    assert len(adv) > 0

    timer[('find', )] = time.time() - start
    return adv, timer


def audit_raft_logs(path: str, n: int):
    def chain_file_func(i):
        return lambda c: join(path, f'{node.fmt_int(i, n-1)}', f'blockchain_{node.fmt_int(i, n-1)}_{node.fmt_int(c, maxc[i])}.jsonl')

    meta_timer = defaultdict(float)

    ccs, lcs, maxc, filesize, fs_measure = {}, {}, {}, -1, True
    checked, adversarial = set([]), set([])
    for i in range(n):
        dir_ = join(path, node.fmt_int(i, n-1))
        good, nc, cc, lc, timer = check_node_integrity(dir_, node.fmt_int(i, n-1), n)
        for key in timer:
            meta_timer[key] += timer[key]
        ccs[i] = cc
        lcs[i] = lc
        maxc[i] = nc

        if good:
            checked.add(i)
            if fs_measure:
                if nc >= 3:
                    filesize = get_file_size(chain_file_func(i)(0))
                    fs_measure = False
                else:
                    filesize = max(filesize, get_file_size(chain_file_func(i)(0)))
        else:
            adversarial.add(i)

    print(checked)
    for i in checked:
        for j in checked:
            if i >= j:
                continue
            consistency, timer = pair_consistency_check(i, j, chain_file_func(i), chain_file_func(j), maxc)

            for key in timer:
                meta_timer[key] += timer[key]

            if not consistency:
                adv, timer = pair_find_adversary(filesize, chain_file_func(i), chain_file_func(
                    j), maxc[i], maxc[j], ccs[i], ccs[j], lcs[i], lcs[j])
                adversarial.update(adv)
                for key in timer:
                    meta_timer[key] += timer[key]

    meta_timer.update(FUNC_TIMER)
    meta_timer = flatten_dict.unflatten(dict(meta_timer))
    meta_timer['adv'] = list(adversarial)
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
