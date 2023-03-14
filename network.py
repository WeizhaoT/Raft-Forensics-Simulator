from __future__ import annotations

from colorama import Fore

import matplotlib
import numpy as np
import heapq
import time
from typing import List
from tqdm import tqdm
import threading

from node import Node, Rtype, Block
from event import Event, Etype, TestEvents
from delays import BaseDelay

matplotlib.use('Agg')


class Network:
    def __init__(self, n, delay_mgr: BaseDelay, tx_retry: int, adversary: None | int = None) -> None:
        assert n > 1

        self.n = n
        self.tx_retry = tx_retry
        self.delay_mgr = delay_mgr
        self.leader: Node = None
        self.dummy = Node(-1, n)
        self.nodes: List[Node] = [Node(id_=i, n=n) for i in range(n)]
        self.quorum = len(self.nodes) // 2 + 1

        self.aid = -1
        self.progress = {i: (0, 0) for i in range(n)}
        self.progress_commit = (0, 0)

        self.leader_fork = False
        self.bad_vote = False

        if adversary is not None:
            self.aid = adversary
            self.adversary = Node(id_=adversary, n=n, adversarial=True)
            self.adversarial_progress = {i: (0, 0) for i in range(n)}
            self.adversarial_commit = (0, 0)
        else:
            self.adversary = None

    def is_leader(self, node: Node):
        return node.id == self.leader.id

    def delay(self, node1, node2):
        if isinstance(node1, Node):
            node1 = node1.id
        if isinstance(node2, Node):
            node2 = node2.id

        return self.delay_mgr(node1, node2)

    def commit(self, evt: Event, adversarial=False):
        if adversarial:
            leader, progress, prog_commit = self.adversary, self.adversarial_progress, self.adversarial_commit
        else:
            leader, progress, prog_commit = self.leader, self.progress, self.progress_commit

        res = []
        ranks = sorted(progress, key=progress.get, reverse=True)
        current_push = progress[ranks[self.quorum - 1]]
        cc = ranks[:self.quorum]

        if current_push > prog_commit:
            if adversarial:
                self.adversarial_commit = current_push
            else:
                self.progress_commit = current_push

            leader.commit(*current_push, cc)
            leader.write_log(evt.t, f'Commit block {str(leader.head)}')
            for node in leader.listeners:
                leader.write_log(evt.t, f'->{node.name} Ask to commit block {str(leader.head)}')
            res = [Event(node, Etype.CMT, evt.t + self.delay(leader, node), *current_push, cc)
                   for node in leader.listeners]
        elif current_push == prog_commit:
            follower, _, _ = evt.args
            leader.write_log(evt.t, f'->{follower.name} Ask to commit block {str(leader.head)}')
            res = [Event(follower, Etype.CMT, evt.t + self.delay(leader, follower), *current_push, cc)]

        return res

    def split_honest(self):
        ns = sorted([n for n in self.nodes if n.id != self.aid], key=lambda n: n.head.h)

        leaf = set([ns[0]])
        prefixes = {}
        for i, n in enumerate(ns[1:]):
            leaf.add(n)
            for j in range(i, -1, -1):
                if n.has(ns[j].head):
                    prefixes[n.id] = ns[j]
                    leaf.discard(ns[j])
                    break

        leaf = list(leaf)
        if len(leaf) > 2:
            raise ValueError('3 branches detected. Cannot split consensus in halves')

        v1, s1 = leaf[0], [leaf[0]]
        while len(s1) < self.quorum - 1 and v1.id in prefixes:
            v1 = prefixes[v1.id]
            s1.append(v1)

        if len(s1) < self.quorum - 1:
            raise ValueError('a branch does not have (n-1)/2 nodes')

        s1_set = set(s1)
        s2 = [n for n in ns[::-1] if n not in s1_set]
        if len(leaf) == 2:
            v2, s2p = leaf[1], [leaf[1]]
            while len(s2p) < self.quorum - 1 and v2.id in prefixes:
                v2 = prefixes[v2.id]
                if v2 not in s1_set:
                    s2p.append(v2)

            if len(s2p) < self.quorum - 1:
                raise ValueError('a branch does not have (n-1)/2 nodes')

        del s1_set

        leader = self.nodes[self.aid]

        leader.steal_from(s1[0])
        leader.clear_listeners()
        leader.add_listeners(s1)

        self.adversary.steal_from(s2[0])
        self.adversary.clear_listeners()
        self.adversary.add_listeners(s2)

        return s1, s2

    def make_lc(self, lid):
        lc, acc, candidate = [lid], [], self.nodes[lid]
        for node in self.nodes:
            if node.id == lid:
                continue

            if candidate.has(node.tail):
                # print(candidate.tail, node.tail)
                if len(lc) < self.quorum:
                    lc.append(node.id)
            if candidate.has(node.head):
                acc.append(node)

        return lc, acc

    def resolve(self, evt: Event):
        if evt.subject is None:
            evt.subject = self.dummy

        if evt.type == Etype.INFO:
            return []
        if evt.type == Etype.SET_FORK:
            self.leader_fork = True
            return []
        elif evt.type == Etype.UNSET_FORK:
            self.leader_fork = False
            return []
        elif evt.type == Etype.AUTOLEAD:
            if self.leader_fork and self.leader.id == self.aid:
                nrank = sorted(self.leader.peers, key=lambda node: node.freshness, reverse=True)
                new_leader = nrank[1].id if nrank[1] != self.aid else nrank[0].id
                self.leader.steal_from(self.nodes[new_leader])
            else:
                nrank = sorted(self.leader.peers, key=lambda node: node.freshness)
                new_leader = nrank[self.quorum - 1].id

            # return self.handle_new_leader(Event(None, Etype.LEAD, evt.t, new_leader))
            return [Event(None, Etype.LEAD, evt.t, new_leader)]
        elif evt.type == Etype.LEAD:
            return self.handle_new_leader(evt)
        elif evt.type == Etype.TX:
            tx, rem = evt.args
            if self.leader.id >= 0:
                # ADVERSARIAL: FORK AS LEADER
                if self.aid == self.leader.id and self.leader_fork and (self.leader.tail.h + self.adversary.tail.h) % 2 == 0:
                    leader = self.adversary
                    block = leader.accept_tx(tx)
                    self.adversarial_progress[leader.id] = (block.t, block.h)
                else:
                    leader = self.leader
                    block = leader.accept_tx(tx)
                    self.progress[leader.id] = (block.t, block.h)

                for node in leader.listeners:
                    leader.write_log(evt.t, f'->{node.name} Replicating {Block.strlist([block])}')

                return [Event(node, Etype.REP, evt.t + self.delay(leader, node), leader, leader.term, [block])
                        for node in leader.listeners]
            else:
                if rem > 0:
                    return [Event(self.dummy, Etype.TX, evt.t + self.tx_retry, tx, rem-1)]
                else:
                    return []
        elif evt.type == Etype.REP:
            leader, term, blocks, = evt.args
            if term < evt.subject.term:
                return []

            follower = evt.subject
            res = follower.handle_append_entries(blocks)
            # if follower.set_acked():
            if res == Rtype.YES:
                if follower.set_acked():
                    follower.write_log(evt.t, f'{leader.name}-> {Block.strlist(blocks)} Replication accepted')
                    return [Event(leader, Etype.ACK, evt.t + self.delay(follower, leader), follower, blocks[-1].t, blocks[-1].h)]
            elif res == Rtype.KEEP:
                if follower.set_asked():
                    follower.write_log(
                        evt.t, f'{leader.name}-> {Block.strlist(blocks)} Replication needs more blocks')
                    return [Event(leader, Etype.R2, evt.t + self.delay(follower, leader), follower, follower.head.h)]
            else:
                follower.write_log(evt.t, f'{leader.name}-> {Block.strlist(blocks)} Replication rejected')
                return []

            return []
            # else:
            #     follower.write_log(
            #         evt.t, f'{leader.name}-> {Block.strlist(blocks)} Replication ignored ({follower.tail.h})')
            #     return []
        elif evt.type == Etype.ACK:
            if not self.is_leader(evt.subject):
                return []

            follower, t, h, = evt.args
            fid = follower.id

            follower.write_log(evt.t, f'->{evt.subject.name} ACK term {t} and height {h}')
            evt.subject.write_log(evt.t, f'{follower.name}-> ACK term {t} and height {h}')
            if evt.subject.adversarial:
                if self.adversarial_progress[fid] < (t, h):
                    self.adversarial_progress[fid] = (t, h)
                    return self.commit(evt, adversarial=True)
            else:
                if self.progress[fid] < (t, h):
                    self.progress[fid] = (t, h)
                    return self.commit(evt, adversarial=False)

            return []
        elif evt.type == Etype.R2:
            if not self.is_leader(evt.subject):
                return []

            follower, fh, = evt.args
            leader = evt.subject
            blocks = leader.read_blocks_since(fh)
            follower.write_log(evt.t, f'->{leader.name} ask for more blocks from height {fh}')
            leader.write_log(evt.t, f'{follower.name}-> send more blocks: {Block.strlist(blocks)}')
            return [Event(follower, Etype.REP, evt.t + self.delay(leader, follower), leader, leader.term, blocks)]
        elif evt.type == Etype.CMT:
            t, h, cc = evt.args
            follower = evt.subject
            res = follower.commit(t, h, cc)
            if res:
                evt.subject.write_log(evt.t, f'Commit block {str(follower.head)}')
            else:
                evt.subject.write_log(evt.t, f'Refuse to commit block at term {t}, height {h}')

            return []

    def run(self, period, maxtime, tx_interval, tx_count, max_tx_retry=20, sleep=0, debug=True):
        # events: List[Event] = TestEvents.TEST_LEADER_CHANGE_2(tx_count, tx_interval, max_tx_retry)
        events: List[Event] = TestEvents.TEST_FORK_1(tx_count, tx_interval, max_tx_retry)
        heapq.heapify(events)

        bar = tqdm(total=maxtime // period)

        ef = None
        if debug:
            ef = open('event.log', 'w')

        lock = threading.Lock()
        global looping
        looping = True

        def proceed(t):
            global looping

            if t + period < maxtime and looping:
                if events:
                    threading.Timer(sleep, proceed, (t + period,)).start()
                else:
                    threading.Timer(0, proceed, (t + period,)).start()

            with lock:
                try:
                    while events and events[0].t <= t:
                        event = heapq.heappop(events)
                        new_events = self.resolve(event)
                        try:
                            assert isinstance(new_events, list)
                        except:
                            print(event)
                            raise

                        for new_event in new_events:
                            heapq.heappush(events, new_event)

                        if debug:
                            ef.write(f'{str(event)}\n')

                    if debug:
                        ef.flush()

                    def brief(node: Node):
                        p = f'{node.tail.h}/{node.head.h}'
                        return Fore.YELLOW + p + Fore.RESET if node == self.leader else p

                    bar.set_description(f'{Fore.LIGHTBLUE_EX}[{t} ms @ {self.progress_commit[1]}]{Fore.RESET} ' +
                                        ' '.join(brief(v) for v in self.nodes))
                    bar.update()

                    if t + period >= maxtime:
                        looping = False
                except:
                    looping = False
                    raise

        proceed(0)
        while looping:
            time.sleep(1)

        bar.close()

        for node in self.nodes:
            node.flush_uncommitted()

        if self.adversary:
            self.adversary.flush_uncommitted()

        if debug:
            ef.close()

    def handle_new_leader(self, evt: Event):
        lid, = evt.args
        if lid < 0:
            self.leader = self.dummy
            return []

        lc, acc = [], []
        term = max(n.term for n in self.nodes) + 1
        if self.leader_fork and lid == self.aid:
            s1, s2 = self.split_honest()
            id1, id2 = list(map(lambda x: x.id, s1)), list(map(lambda x: x.id, s2))
            self.leader = self.nodes[lid]
            prep = [(self.leader, s1, term, id1, self.progress),
                    (self.adversary, s2, term+1, id2, self.adversarial_progress)]
        else:
            lc, acc = self.make_lc(lid)
            if len(lc) < self.quorum:
                return []

            self.leader = self.nodes[lid]
            prep = [(self.leader, acc, term, lc, self.progress)]

        for args in prep:
            self.prepare_for_new_leader(*args)

        return [Event(None, Etype.INFO, evt.t, "fork", id1, id2)] if (self.leader_fork and lid == self.aid) else []

    def prepare_for_new_leader(self, leader: Node, followers: List[Node], term: int, lc: List[int], progress):
        leader.clear_listeners()
        leader.add_listeners(followers)
        leader.update_term(term)

        ft, fh = leader.freshness
        leader.accept_leader(term, ft, fh, leader.id, lc)

        for node in followers:
            node.accept_leader(term, ft, fh, leader.id, lc)

        progress.clear()
        progress.update({f.id: (0, 0) for f in followers})
        progress[leader.id] = ft, fh

        for node in followers:
            node.reset_leader_prog()
            node.update_term(term)
