from __future__ import annotations

import matplotlib
import numpy as np
import heapq
import time
from typing import List
from tqdm import tqdm
import threading

from node import Node, Rtype
from event import Event, Etype, TestEvents
from delays import BaseDelay

matplotlib.use('Agg')


class Network:
    def __init__(self, n, delay_mgr: BaseDelay, tx_retry, adversary=None) -> None:
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

    def delay(self, node1, node2):
        if isinstance(node1, Node):
            node1 = node1.id
        if isinstance(node2, Node):
            node2 = node2.id

        return self.delay_mgr(node1, node2)

    def commit(self, evt: Event):
        res = []
        ranks = sorted(self.progress, key=self.progress.get, reverse=True)
        current_push = self.progress[ranks[self.quorum - 1]]
        cc = ranks[:self.quorum]

        # print(current_push, self.progress)

        if current_push > self.progress_commit:
            self.progress_commit = current_push
            self.leader.commit(*current_push, cc)
            res = [Event(node, Etype.CMT, evt.t + self.delay(self.leader, node), *current_push, cc)
                   for node in self.leader.listeners]

        if not self.leader_fork:
            return res

        # TODO:

    def split_honest(self):
        assert self.leader.id == self.aid

        for node in self.nodes:
            if not self.leader.has(node.head):
                self.leader.remove_listener(node)
                if self.leader_fork:
                    self.adversary.add_listeners([node])
                    # TODO:

        self.progress

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

        if evt.type == Etype.LEAD:
            return self.handle_new_leader(evt)
        elif evt.type == Etype.TX:
            tx, rem = evt.args
            if self.leader.id >= 0:
                block = self.leader.accept_tx(tx)
                self.progress[self.leader.id] = (block.t, block.h)
                # ADVERSARIAL: FORK AS LEADER
                if self.aid == self.leader.id and self.leader_fork:
                    ablock = self.adversary.accept_tx(tx)
                    self.adversarial_progress[self.aid] = (ablock.t, ablock.h)
                    return [Event(node, Etype.REP, evt.t + self.delay(self.leader, node), self.leader, [block])
                            for node in self.leader.listeners] + \
                        [Event(node, Etype.REP, evt.t + self.delay(self.leader, node), self.adversary, [ablock])
                         for node in self.adversary.listeners]
                else:
                    return [Event(node, Etype.REP, evt.t + self.delay(self.leader, node), self.leader, [block])
                            for node in self.leader.listeners]
            else:
                if rem > 0:
                    return [Event(self.dummy, Etype.TX, evt.t + self.tx_retry, tx, rem-1)]
                else:
                    return []
        elif evt.type == Etype.REP:
            leader, blocks, = evt.args
            if evt.subject.id == self.leader.id:
                return []
            res = evt.subject.handle_append_entries(blocks)
            if res == Rtype.YES:
                return [Event(leader, Etype.ACK, evt.t + self.delay(evt.subject, leader), evt.subject, blocks[-1].t, blocks[-1].h)]
            elif res == Rtype.KEEP:
                return [Event(leader, Etype.R2, evt.t + self.delay(evt.subject, leader), evt.subject, evt.subject.head.h)]
            # elif res == Rtype.REJ and self.leader.id == self.aid:
            #     leader.listeners.remove(evt.subject)
            #     # TODO:
            else:
                return []
        elif evt.type == Etype.ACK:
            if evt.subject.id != self.leader.id:
                return []

            follower, t, h, = evt.args
            fid = follower.id
            if evt.subject.adversarial:
                self.adversarial_progress[fid] = max(self.adversarial_progress[fid], (t, h))
            else:
                self.progress[fid] = max(self.progress[fid], (t, h))
            return self.commit(evt)
        elif evt.type == Etype.R2:
            if evt.subject != self.leader:
                return []
            follower, fh, = evt.args
            return [Event(follower, Etype.REP, evt.t + self.delay(self.leader, follower), self.leader, self.leader.read_blocks_since(fh))]
        elif evt.type == Etype.CMT:
            t, h, cc = evt.args
            res = evt.subject.commit(t, h, cc)
            # if not res:
            #     print(evt.subject.id, evt.subject.head, evt.subject.tail, t, h, cc)
            return []

    def run(self, period, maxtime, tx_interval, tx_count, max_tx_retry=20, sleep=0, debug=True):
        events: List[Event] = TestEvents.TEST_LEADER_CHANGE_2(tx_count, tx_interval, max_tx_retry)
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

                        for new_event in new_events:
                            heapq.heappush(events, new_event)

                        if debug:
                            ef.write(f'{str(event)}\n')

                    if debug:
                        ef.flush()

                    bar.set_description(f'[{t} ms @ {self.progress_commit[1]}] ' +
                                        ' '.join(f'{v.tail.h}/{v.head.h}' for p, v in zip(self.progress, self.nodes)))
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

        if debug:
            ef.close()

    def handle_new_leader(self, evt: Event):
        lid, = evt.args
        if lid < 0:
            self.leader = self.dummy
            return []

        lc, acc = [], []
        if self.leader_fork:
            lid = self.aid
        else:
            lc, acc = self.make_lc(lid)
            if len(lc) < self.quorum:
                return []

        self.delay_mgr.rebase(lid)

        self.leader = self.nodes[lid]
        self.leader.clear_listeners()
        self.leader.add_listeners(acc)
        term = self.leader.increment_term()

        ft, fh = self.leader.tail.t, self.leader.tail.h
        self.leader.accept_leader(term, ft, fh, lid, lc)

        for node in acc:
            node.accept_leader(term, ft, fh, lid, lc)

        self.progress = {i: (0, 0) for i in range(self.n)}
        self.progress[lid] = ft, fh

        for node in self.nodes:
            if node.id == lid:
                continue

            node.update_term(term)

        if self.leader_fork:
            self.split_honest()

        return []
