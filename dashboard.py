from utils import *
from sql import *
import os
import time
import argparse
import json
import datetime
EXP=''
COMMIT={}
CONFLICT=False

class Synchronizer(object):
    
    def __init__(self, id):
        self.id = id
        self.chainfile = [f"{EXP}/{id}/blockchain_{id}_0.jsonl", 0]
        self.commitfile = [f"{EXP}/{id}/commitment_{id}.jsonl", 0]
        self.leaderfile = [f"{EXP}/{id}/leader_{id}.jsonl", 0]
        self.height = 0

    def syncall(self):
        self.sync_chain()
        self.sync_commit()
        self.sync_leader()

    def sync_chain(self):
        global CONFLICT, COMMIT
        filename, t = self.chainfile
        if not os.path.isfile(filename): return
        stamp = os.stat(filename).st_mtime
        if stamp != t:
            self.chainfile[1] = stamp
            print('update chain', self.id)
            with open(filename, 'r') as json_file:
                json_list = list(json_file)
                
                for i in range(self.height, len(json_list)):
                    latest = json.loads(json_list[i])
                    insert_node(f'blockchain_{self.id}', 
                                (latest['t'], latest['h'], latest['pt'], 'tx' + latest['tx']))
                    h = latest['h']
                    t = latest['tx']
                    if h not in COMMIT: COMMIT[h] = t
        
                    if COMMIT[h] != t:  
                        CONFLICT = True
                        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                        insert_conflict((timestamp, latest['t'], h, 1))
                
                self.height = len(json_list)
     
    def sync_commit(self):
        filename, t = self.commitfile
        if not os.path.isfile(filename): return
        stamp = os.stat(filename).st_mtime
        if stamp != t:
            self.commitfile[1] = stamp
            print('update commit', self.id)
            with open(filename, 'r') as json_file:
                json_list = list(json_file)
                latest = json.loads(json_list[-1])
                insert_cc(f'cc_{self.id}', 
                            (latest['t'], latest['h'], str(latest['voters'])))
            

    def sync_leader(self):
        filename, t = self.leaderfile
        if not os.path.isfile(filename): return
        stamp = os.stat(filename).st_mtime
        if stamp != t:
            self.leaderfile[1] = stamp
            print('update leader', self.id)
            with open(filename, 'r') as json_file:
                json_list = list(json_file)
                latest = json.loads(json_list[-1])
                insert_lc(f'lc_{self.id}', 
                            (latest['t'], latest['leader'], str(latest['voters'])))

def audit():
    with open(f'{EXP}/audit.json', 'r') as json_file:
        info = json.load(json_file)
        adv = info['adv']
        evidence = info['evidence']
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        insert_text((timestamp, f'culprit: {adv}\nevidence: {evidence}'))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--period', type=int, default=50, help='period')
    parser.add_argument('-M', '--maxtime', type=int, default=10000, help='max time')
    parser.add_argument('-s', '--seed', type=int, default=120, help='Random Seed')
    parser.add_argument('-e', '--exp', type=str, default='', help='experiment name')
    args = parser.parse_args()

    EXP = args.exp

    n = 5
    synchronizers = [Synchronizer(i) for i in range(n)]
    clear_conflict()
    clear_text()
    for i in range(n):
        clear_node(f"blockchain_{i}")
        clear_cc(f"cc_{i}")
        clear_lc(f"lc_{i}")
        
        
    while True:
        if CONFLICT: break
        for i in range(n):
            if CONFLICT: 
                print('halt due to conflict')
                log = os.popen(f'python audit.py -p {EXP}').read()
                print(log)
                audit()
                break
            synchronizers[i].syncall()
        time.sleep(2)
