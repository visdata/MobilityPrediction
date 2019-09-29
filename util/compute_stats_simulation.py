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

input_dir = '/datahouse/yurl/TalkingData/data/BJ-simulation-inputdata/'
output_dir = '/datahouse/yurl/TalkingData/data/BJ-simulation-inputdata/'

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, dest='filename',
                    help='(required) the filename threshold', required=True)
args = parser.parse_args()

filename = args.filename

def convert_to_hour(seconds):
    hour = int((seconds - 1467000000) / 3600) % (7 * 24)
    return hour

def convert_longitude(data, split):
    return int((data - 115.422) / split)

def convert_latitude(data, split):
    return int((data - 39.445) / split)

def distance(lat1, lon1, lat2, lon2):
    """
    compute distance given two points
    """
    # radius of the earth by km
    RADIUS_EARTH = 6371
    DEGREE_TO_RADIAN = 2 * math.pi / 360
    COS_LATITUDE = 0.77

    lat1 = lat1 * DEGREE_TO_RADIAN
    lon1 = lon1 * DEGREE_TO_RADIAN
    lat2 = lat2 * DEGREE_TO_RADIAN
    lon2 = lon2 * DEGREE_TO_RADIAN
    x = (lon2 - lon1) * COS_LATITUDE
    y = lat2 - lat1
    return int(RADIUS_EARTH * sqrt(x * x + y * y) * 1000)


def compute_sparsity(filename):
    """
    label the file given the filename
    append 0 after the stay record, append 1 after the travel record, do nothing for other records
    """
    start_time = time.time()

    filename_r = input_dir + filename
    filename_w_tjt = output_dir + filename + '-trajectory'
    filename_w_sparsity = output_dir + filename + '-sparsity'
    
    with open(filename_r) as f:
        records = f.readlines()

    c_uid = -1
    segments, tjt = [], []
    trajectorys, stats = [], []
    stay_num, travel_num = 0, 0
    
    # divide the records into to segments
    for record in records:
        columns = record.split(',')

        if len(columns) < 5:
            print('An error line in line: ' + str(record))
            continue

        # set record columns
        uid = columns[0]
        time_second = int(columns[1])
        latitude, longtitude, state = float(columns[2]), float(columns[3]), int(columns[4])
        
        if state == 0:
            stay_num = stay_num + 1
        else:
            travel_num = travel_num + 1

        # check if it is the same trajectory
        if uid == c_uid:
            tjt.append([uid, time_second, latitude, longtitude, state])
        else:
            # new uid
            if c_uid != -1:
                # the current uid is valid, segment the trajectory of the current uid (c_uid)
				# sort the trajectory by time
                tjt.sort(key=lambda x: x[1])
                

                # store results
                
                if len(tjt) > 0:
                    trajectorys.append(tjt)
                    stats.append([c_uid, float(0), float(0), stay_num, travel_num, len(tjt)])

                # reset
                tjt = []
                stay_num, travel_num = 0, 0

            
            # refresh the arrays to only store the first record of the new trajectory (uid)
            tjt.append([uid, time_second, latitude, longtitude, state])
            c_uid = uid
    
    # output to file
    with open(filename_w_tjt, 'w') as f:
        for trajectory in trajectorys:
            segments = [','.join([str(x) for x in record]) for record in trajectory]
            f.write('|'.join(segments) + '\n')
    with open(filename_w_sparsity, 'w') as f:
        for stat in stats:
            f.write(','.join([str(x) for x in stat]) + '\n')
    
    stay_num, travel_num, all_num = sum([x[3] for x in stats]), sum([x[4] for x in stats]), sum([x[5] for x in stats])
    print('[file %s] time %f, records num %d, stay num %d (%f%%), travel num %d (%f%%)'
        %(filename, time.time() - start_time, all_num, stay_num, stay_num / all_num * 100, travel_num, travel_num / all_num * 100))

if __name__ == "__main__":
#     pool = multiprocessing.Pool(processes=15)
#     pool.map(label_and_compute_sparsity, filelist)
    compute_sparsity(filename);
