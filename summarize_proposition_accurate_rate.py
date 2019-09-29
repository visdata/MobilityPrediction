"""
the edge of the period
'beijing': {
      'north':  41.055,
      'south':  39.445,
      'west':  115.422,
      'east':  117.515
    }
Time Reference:
1467000000: Mon 12:00:00 2016-6-27
"""
from __future__ import print_function, division

from math import sin, cos, sqrt, atan2, radians
import time
import math
import sys
import argparse
import multiprocessing

        
input_path = '/datahouse/yurl/TalkingData/data/P3-SS-BJ/'
output_filename = '/datahouse/yurl/TalkingData/data/P3-SS-BJ/overall_stats/proposition-count.csv'

parser = argparse.ArgumentParser()
parser.add_argument('--minute', type=int, dest='minute',
                    help='(required) the time threshold, unit: minute, e.g. 15', required=True)
parser.add_argument('--space', type=int, dest='space',
                    help='(required) the space threshold, unit: meter, e.g. 800', required=True)
args = parser.parse_args()

minute = args.minute
space = args.space

filelist = ['part-'+format(n, '05d') for n in range(10000) if (n%100==0)]

pos_num, neg_num = 0, 0;

for filename in filelist:
    try:
        with open(input_path+filename+'-proposition-count_' + str(minute) + "-" + str(space), 'r') as f:
            columns = f.readline().strip().split(',')
            pos_num = pos_num + long(columns[0])
            neg_num = neg_num + long(columns[1])
    except IOError:
        continue

with open(output_filename, 'a') as f:
    f.write(','.join([str(minute),str(space),str(pos_num), str(neg_num)]) + '\n')
      
