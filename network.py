from __future__ import annotations

import matplotlib
import numpy as np
import heapq
import time
from typing import List
from tqdm import tqdm

from node import Node, F_NO, F_YES, F_KEEP
from event import Event, Etype
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
        self.progress = [(0, 0)] * n
        self.progress_commit = (0, 0)

        self.leader_fork = False
        self.bad_vote = False

        if adversary is not None:
            self.aid = adversary
            self.adversary = Node(id_=adversary, n=n, adversarial=True)
            self.adversarial_progress = [(0, 0)] * n
            self.adversarial_commit = (0, 0)

    def delay(self, node1, node2):
        if isinstance(node1, Node):
            node1 = node1.id
        if isinstance(node2, Node):
            node2 = node2.id

        return self.delay_mgr(node1, node2)

    def commit(self, evt: Event):
        current_push = sorted(self.progress)[self.quorum - 1]
        if current_push > self.progress_commit:
            self.progress_commit = current_push
            self.leader.commit(*current_push)
            return [Event(node, Etype.CMT, evt.t + self.delay(self.leader, node), *current_push) for node in self.nodes if node != self.leader]
        else:
            return []

    def split_honest(self):
        self.progress

    def resolve(self, evt: Event):
        if evt.type == Etype.LEAD:
            lid, = evt.args
            if lid < 0:
                self.leader = self.dummy
                return []

            if self.leader_fork:
                lid = self.aid

            self.delay_mgr.rebase(lid)
            self.leader = self.nodes[lid]
            term = self.leader.increment_term()
            for node in self.nodes:
                node.update_term(term)

            return []
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
                            for node in self.nodes if node != self.leader] + \
                        [Event(node, Etype.REP, evt.t + self.delay(self.leader, node), self.adversary, [ablock])
                         for node in self.nodes if node != self.leader]
                else:
                    return [Event(node, Etype.REP, evt.t + self.delay(self.leader, node), self.leader, [block])
                            for node in self.nodes if node != self.leader]
            else:
                if rem > 0:
                    return [Event(self.dummy, Etype.TX, evt.t + self.tx_retry, tx, rem-1)]
                else:
                    return []
        elif evt.type == Etype.REP:
            leader, blocks, = evt.args
            res = evt.subject.handle_append_entries(blocks)
            if res == F_YES:
                return [Event(leader, Etype.ACK, evt.t + self.delay(evt.subject, leader), evt.subject, blocks[-1].t, blocks[-1].h)]
            elif res == F_KEEP:
                return [Event(leader, Etype.R2, evt.t + self.delay(evt.subject, leader), evt.subject, evt.subject.head.h)]
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
            t, h = evt.args
            evt.subject.commit(t, h)
            return []

    def run(self, period, maxtime, tx_interval, tx_count, max_tx_retry=20, sleep=0, debug=True):
        events: List[Event] = [Event(self.dummy, Etype.LEAD, 0, 0)] + \
            [Event(self.dummy, Etype.TX, tc * tx_interval, f'{tc:4d}', max_tx_retry) for tc in range(1, tx_count+1)] +\
            []  # [Event(self.dummy, Etype.LEAD, 1000, -1), Event(self.dummy, Etype.LEAD, 1100, 3)]
        heapq.heapify(events)

        bar = tqdm(total=maxtime // period)

        ef = None
        if debug:
            ef = open('event.log', 'w')

        for t in range(0, maxtime, period):
            while events and events[0].t <= t:
                event = heapq.heappop(events)
                try:
                    new_events = self.resolve(event)
                    assert new_events is not None
                except:
                    print(event)
                    raise
                for new_event in new_events:
                    heapq.heappush(events, new_event)

                if debug:
                    ef.write(f'{str(event)}\n')

            if debug:
                ef.flush()
            if events:
                time.sleep(sleep)

            bar.set_description(f'[{t} ms @ {self.progress_commit[1]}] ' +
                                ' '.join(f'{v.tail.h}/{v.head.h}' for p, v in zip(self.progress, self.nodes)))
            bar.update()

        bar.close()

        if debug:
            ef.close()
