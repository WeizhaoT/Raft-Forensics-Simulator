from __future__ import annotations

from colorama import Fore

import os
import shutil
import heapq
import time
from typing import List, Tuple
from tqdm import tqdm
import threading
from collections import defaultdict

import node
import event
import delays


class Network:
    def __init__(
        self,
        n,
        delay_mgr: delays.BaseDelay,
        tx_retry_time: int,
        datadir: str,
        debug: bool,
        adversary: None | int = None
    ) -> None:
        assert n > 1

        if os.path.exists(datadir):
            shutil.rmtree(datadir)

        def nodedir(i, adv=False):
            return os.path.join(datadir, ('a' if adv else '') + node.fmt_int(i, n-1))

        os.makedirs(datadir)
        for i in range(n):
            os.makedirs(nodedir(i))

        self.n = n
        self.tx_retry = tx_retry_time
        self.delay_mgr = delay_mgr
        self.leader: node.Node = None
        self.dummy = node.Node(-1, n)
        self.nodes: List[node.Node] = [node.Node(id_=i, n=n, dir_=nodedir(i)) for i in range(n)]
        self.quorum = len(self.nodes) // 2 + 1

        self.aid = -1
        self.progress = {i: (0, 0) for i in range(n)}
        self.progress_commit = (0, 0)

        self.leader_fork = False
        self.bad_vote = False

        self.DEBUG = debug

        if adversary is not None:
            os.makedirs(nodedir(adversary, True))
            self.aid = adversary
            self.adversary = node.Node(id_=adversary, n=n, dir_=nodedir(adversary, True), adversarial=True)
            self.adversarial_progress = {i: (0, 0) for i in range(n)}
            self.adversarial_commit = (0, 0)
        else:
            self.adversary = None

    @property
    def forking(self) -> bool:
        return self.leader_fork and self.leader and self.leader.id == self.aid

    def is_leader(self, v: node.Node):
        return v.id == self.leader.id

    def delay(self, node1, node2):
        if isinstance(node1, node.Node):
            node1 = node1.id
        if isinstance(node2, node.Node):
            node2 = node2.id

        return self.delay_mgr(node1, node2)

    def commit(self, evt: event.Event, adversarial=False):
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

            if self.DEBUG:
                leader.write_log(evt.t, f'Commit block {str(leader.head)}, CC = {cc}')
                for v in leader.listeners:
                    leader.write_log(evt.t, f'->{v.name} Ask to commit block {str(leader.head)}')

            res = [event.Event(v, event.Etype.CMT, evt.t + self.delay(leader, v), leader, *current_push, cc)
                   for v in leader.listeners]
        elif current_push == prog_commit:
            follower, _, _ = evt.args

            if self.DEBUG:
                leader.write_log(evt.t, f'->{follower.name} Ask to commit block {str(leader.head)}')

            res = [event.Event(follower, event.Etype.CMT, evt.t +
                               self.delay(leader, follower), leader, *current_push, cc)]

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

    def make_lc(self, lid) -> Tuple[List[int], List[node.Node]]:
        lc, acc, candidate = [lid], [], self.nodes[lid]
        for v in self.nodes:
            if v.id == lid:
                continue

            if candidate.has(v.tail):
                if len(lc) < self.quorum:
                    lc.append(v.id)
            if candidate.has(v.head):
                acc.append(v)

        return lc, acc

    def resolve(self, evt: event.Event):
        if evt.subject is None:
            evt.subject = self.dummy

        if evt.type == event.Etype.INFO:
            return []
        if evt.type == event.Etype.SET_FORK:
            self.leader_fork = True
            return []
        elif evt.type == event.Etype.UNSET_FORK:
            self.leader_fork = False
            return []
        elif evt.type == event.Etype.AUTOLEAD:
            if self.forking:
                nrank = sorted(self.leader.peers, key=lambda v: v.freshness, reverse=True)
                new_leader = nrank[1].id if nrank[1] != self.aid else nrank[0].id
                self.leader.steal_from(self.nodes[new_leader])
            elif self.leader_fork:  # or (self.bad_vote and self.leader.id != self.aid):
                new_leader = self.aid
                self.nodes[self.aid].steal_from(self.leader)
            else:
                nrank = sorted(self.leader.peers, key=lambda v: v.freshness)
                new_leader = nrank[self.quorum - 1].id
                if new_leader == self.leader.id:
                    new_leader = nrank[-1].id

            return [event.Event(None, event.Etype.LEAD, evt.t, new_leader)]
        elif evt.type == event.Etype.FORK:
            self.leader_fork = True
            return self.handle_new_leader(event.Event(None, event.Etype.LEAD, evt.t, self.aid))
        elif evt.type == event.Etype.BADVOTELEAD:
            nrank = sorted(self.leader.peers, key=lambda v: v.freshness, reverse=True)
            worst = len(nrank) - self.quorum
            if self.nodes[self.aid].freshness <= nrank[worst + 1].freshness:
                return [event.Event(None, event.Etype.BADVOTELEAD, evt.t + self.tx_retry)]
            if nrank[worst + 1].has(nrank[worst].head):
                return [event.Event(None, event.Etype.BADVOTELEAD, evt.t + self.tx_retry)]

            new_leader = nrank[-self.quorum + 1].id
            if new_leader == self.aid:
                new_leader = nrank[-self.quorum + 2].id

            self.nodes[self.aid].steal_from(self.nodes[new_leader])
            if self.DEBUG:
                self.nodes[self.aid].write_log(evt.t, f'casted a bad vote for {new_leader}')
            self.bad_vote = True
            print(new_leader, ', '.join(f'{i.name}: {i.freshness}/({i.head.t}, {i.head.h})' for i in nrank))
            return [event.Event(None, event.Etype.LEAD, evt.t, new_leader)]
        elif evt.type == event.Etype.LEAD:
            return self.handle_new_leader(evt)
        elif evt.type == event.Etype.TX:
            tx, rem = evt.args
            if self.leader.id >= 0:
                # ADVERSARIAL: FORK AS LEADER
                if self.forking and (self.leader.tail.h + self.adversary.tail.h) % 2 == 0:
                    leader = self.adversary
                    block = leader.accept_tx(tx)
                    self.adversarial_progress[leader.id] = (block.t, block.h)
                else:
                    leader = self.leader
                    block = leader.accept_tx(tx)
                    self.progress[leader.id] = (block.t, block.h)

                if self.DEBUG:
                    for v in leader.listeners:
                        leader.write_log(evt.t, f'->{v.name} Replicating {node.Block.strlist([block])}')

                return [event.Event(v, event.Etype.REP, evt.t + self.delay(leader, v), leader, leader.term, [block])
                        for v in leader.listeners]
            else:
                if rem > 0:
                    return [event.Event(self.dummy, event.Etype.TX, evt.t + self.tx_retry, tx, rem-1)]
                else:
                    return []
        elif evt.type == event.Etype.REP:
            leader, term, blocks, = evt.args
            if term < evt.subject.term or len(blocks) == 0:
                return []

            follower = evt.subject
            res = follower.handle_append_entries(blocks)
            if res == node.Rtype.YES:
                if follower.set_acked():
                    if self.DEBUG:
                        follower.write_log(evt.t, f'{leader.name}-> {node.Block.strlist(blocks)} Replication accepted')
                    return [event.Event(leader, event.Etype.ACK, evt.t + self.delay(follower, leader), follower, blocks[-1].t, blocks[-1].h)]
            elif res == node.Rtype.KEEP:
                if follower.set_asked(blocks[-1].h):
                    if self.DEBUG:
                        follower.write_log(
                            evt.t, f'{leader.name}-> {node.Block.strlist(blocks)} Replication needs more blocks')
                    return [event.Event(leader, event.Etype.R2, evt.t + self.delay(follower, leader), follower, follower.head.h)]
            elif self.DEBUG:
                if res == node.Rtype.REJ1:
                    follower.write_log(
                        evt.t, f'{leader.name}-> Rejected (block/head mismatch) {node.Block.strlist(blocks)}')
                elif res == node.Rtype.REJ3:
                    follower.write_log(
                        evt.t, f'{leader.name}-> Rejected (bad chain) {node.Block.strlist(blocks)}')
                elif res == node.Rtype.REJ2:
                    follower.write_log(
                        evt.t, f'{leader.name}-> Rejected (diff chain mismatch) {node.Block.strlist(blocks)}')

            return []
        elif evt.type == event.Etype.ACK:
            follower, t, h, = evt.args
            fid = follower.id

            if not self.is_leader(evt.subject) or t < evt.subject.term or not follower in evt.subject.listeners:
                return []

            if self.DEBUG:
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
        elif evt.type == event.Etype.R2:
            follower, fh, = evt.args
            leader = evt.subject
            if not self.is_leader(evt.subject) or not follower in leader.listeners:
                return []

            blocks = leader.read_blocks_since(fh)
            if self.DEBUG:
                follower.write_log(evt.t, f'->{leader.name} ask for more blocks from height {fh}')
                leader.write_log(evt.t, f'{follower.name}-> send more blocks: {node.Block.strlist(blocks)}')
            return [event.Event(follower, event.Etype.REP, evt.t + self.delay(leader, follower), leader, leader.term, blocks)]
        elif evt.type == event.Etype.CMT:
            leader, t, h, cc = evt.args
            follower = evt.subject
            if not self.is_leader(leader) or not follower in leader.listeners:
                return []

            res = follower.commit(t, h, cc)
            if self.DEBUG:
                if res == node.Rtype.YES:
                    evt.subject.write_log(evt.t, f'Commit block {str(follower.head)}, CC = {cc}')
                elif res == node.Rtype.REJ1:
                    evt.subject.write_log(evt.t, f'Refuse to commit block at term {t}, height {h}')
                elif res == node.Rtype.NO:
                    evt.subject.write_log(evt.t, f'Block-to-commit is old at term {t}, height {h}')
                elif res == node.Rtype.KEEP:
                    evt.subject.write_log(evt.t, f'Block-to-commit is too new at term {t}, height {h}')

            return []

    def run(self, period: int, maxtime: int, events: List[event.Event], plans: List[event.EventPlan], sleep: int = 0):
        heapq.heapify(events)
        maxt_plans = max(plan.maxt for plan in plans)

        bar = tqdm(total=maxtime // period)

        ef = open('event.log', 'w') if self.DEBUG else None

        lock = threading.Lock()
        global looping
        looping = True

        t = 0
        while t < maxtime:
            for plan in plans:
                planned = plan.events_until(t)
                for evt in planned:
                    heapq.heappush(events, evt)

            # timer = defaultdict(float)
            while events and events[0].t <= t:
                event = heapq.heappop(events)
                # t0 = time.time()
                new_events = self.resolve(event)
                # timer[event.type.name] += time.time() - t0

                try:
                    assert isinstance(new_events, list)
                except:
                    print(event)
                    raise

                for new_event in new_events:
                    heapq.heappush(events, new_event)

                if self.DEBUG:
                    ef.write(f'{str(event)}\n')

            if self.DEBUG:
                ef.flush()

            # consumer = max(timer, key=lambda k: timer[k])

            prog = f'{self.progress_commit[1]}||{Fore.LIGHTRED_EX}{self.adversarial_commit[1]}{Fore.LIGHTBLUE_EX}' if self.forking else f'{self.progress_commit[1]}'

            bar.set_description(f'{Fore.LIGHTRED_EX if self.bad_vote else Fore.LIGHTBLUE_EX}' +
                                f'[{node.ms_to_str(t)} @ {prog}]{Fore.RESET} ' +
                                ' '.join(self.brief(v) for v in self.nodes))
            bar.update()
            t += period

            if events or t < maxt_plans:
                time.sleep(sleep)

        bar.close()

        for v in self.nodes:
            v.flush_uncommitted()

        if self.adversary:
            self.adversary.flush_uncommitted()

        if events:
            print(f'Warning: {len(events)} events not processed')

        if self.DEBUG:
            ef.close()

    def handle_new_leader(self, evt: event.Event):
        lid, = evt.args
        if lid < 0:
            self.leader = self.dummy
            return []

        lc, acc = [], []
        term = max(n.term for n in self.nodes) + 1
        to_fork = self.leader_fork and lid == self.aid
        if to_fork:
            s1, s2 = self.split_honest()
            id1, id2 = list(map(lambda x: x.id, s1)), list(map(lambda x: x.id, s2))
            self.leader = self.nodes[lid]
            prep = [(self.leader, s1, term, [self.aid] + id1, self.progress),
                    (self.adversary, s2, term, [self.aid] + id2, self.adversarial_progress)]
        else:
            lc, acc = self.make_lc(lid)
            if len(lc) < self.quorum:
                return []

            self.leader = self.nodes[lid]
            prep = [(self.leader, acc, term, lc, self.progress)]

        for args in prep:
            leader, acc, term, lc, _ = args
            self.prepare_for_new_leader(*args)
            if self.DEBUG:
                leader.write_log(evt.t, f'Becoming leader of term {term}; accepting followers '
                                 f'{list(map(lambda x: x.id, acc))}, LC = {lc}')
                for f in acc:
                    f.write_log(evt.t, f'Accepting leader of term {term} {leader.name} '
                                f'{list(map(lambda x: x.id, acc))}, LC = {lc}')

        return [event.Event(None, event.Etype.INFO, evt.t, "fork", id1, id2)] if to_fork else []

    def prepare_for_new_leader(self, leader: node.Node, followers: List[node.Node], term: int, lc: List[int], progress):
        leader.clear_listeners()
        leader.add_listeners(followers)
        leader.update_term(term)

        ft, fh = leader.freshness
        leader.accept_leader(term, ft, fh, leader.id, lc)

        for v in followers:
            v.accept_leader(term, ft, fh, leader.id, lc)

        progress.clear()
        progress.update({f.id: (0, 0) for f in followers})
        progress[leader.id] = ft, fh

        self.delay_mgr.rebase(leader.id)

        for v in followers:
            v.reset_leader_prog()
            v.update_term(term)

    def brief(self, v: node.Node):
        if v == self.leader:
            p = f'{v.tail.h}/{v.head.h}'
            if self.forking:
                pa = f'{self.adversary.tail.h}/{self.adversary.head.h}'
                return f'{Fore.YELLOW}{p}{Fore.RESET}||{Fore.RED}{pa}{Fore.RESET}'
            else:
                return f'{Fore.YELLOW}{p}{Fore.RESET}'
        elif self.leader.id < 0:
            return p
        else:
            if self.forking and v in self.adversary.listeners:
                d1, d2 = (v.tail.h - self.adversary.tail.h), (v.head.h - self.adversary.head.h)
                q = f'{d1 if d1 < 0 else "+" + str(d1):>3}/{d2 if d2 < 0 else "+" + str(d2):<3}'
                return f'{Fore.LIGHTRED_EX}{q}{Fore.RESET}'
            else:
                d1, d2 = (v.tail.h - self.leader.tail.h), (v.head.h - self.leader.head.h)
                q = f'{d1 if d1 < 0 else "+" + str(d1):>3}/{d2 if d2 < 0 else "+" + str(d2):<3}'
                return q if v in self.leader.listeners else f'{Fore.LIGHTRED_EX}{q}{Fore.RESET}'
