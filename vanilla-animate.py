import os
import heapq
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from network import Network

matplotlib.use("Agg")
matplotlib.rcParams['font.family'] = ['monospace']

WIDTH = 10
YDISTR = 1.8
PAD = 1
XSPACE = [1, 1, 1, .5, 4]

TEXTPAD = .2
HEAD = .4
SCALE = 2
RADIUS = 0.7
FONTSIZE = 15
LW = 2.5
ALW = 1
ARROWSTYLE = "Simple,head_width=5,head_length=5"

COLOR_DFT = (.7, .7, .7)
COLOR_MSG = (.1, .3, .6)
COLOR_ANN = (.7, .1, .2)
COLOR_TEXT = (0, 0, 0)


def main(pause, framelength, frametime, animationtime, framedir):
    net = Network(7, 0, (60, 80), (200, 300), 160, 1, framedir)
    net.run(framelength, animationtime, frametime, 'vanilla')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pause', type=int, default=5000, help='time in ms to pause')
    parser.add_argument('-l', '--framelength', type=int, default=10, help='time span of each frame')
    parser.add_argument('-f', '--frametime', type=int, default=100, help='frametime in ms')
    parser.add_argument('-a', '--animationtime', type=int, default=1000, help='frametime in ms')
    parser.add_argument('-d', '--dir', type=str, default='vanilla-frames', help='')
    parser.add_argument('-s', '--seed', type=int, default=120, help='Random Seed')

    args = parser.parse_args()

    np.random.seed(args.seed)

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
    else:
        for file in os.listdir(args.dir):
            if 'Frame' in file and file[-4:] == '.jpg':
                os.remove(os.path.join(args.dir, file))

    main(args.pause, args.framelength, args.frametime, args.animationtime, args.dir)
