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

filelist = ['part-'+format(n, '05d') for n in range(10000) if (n%10==0)]

input_path = '/datahouse/yurl/TalkingData/data/BJ_cleaned_data/'
output_path = '/datahouse/yurl/TalkingData/data/P3-SS-BJ/'

parser = argparse.ArgumentParser()
parser.add_argument('--minute', type=int, dest='minute',
                    help='(required) the time threshold, unit: minute, e.g. 15', required=True)
parser.add_argument('--space', type=int, dest='space',
                    help='(required) the space threshold, unit: meter, e.g. 800', required=True)
parser.add_argument('--write_mode', type=int, dest='write_mode', default=1,
                    help='(optional) the output mode (default 1): 0 - write the records line by line, 1 - write all the records of one uid to one line and split the record with |')
args = parser.parse_args()

minute = args.minute
space = args.space
write_mode = args.write_mode

MAX_SPACE_INTERVAL = float(space)
MIN_TIME_INTERVAL = float(minute) * 60
SPLIT = 0.001

MAX_STAY_TRIP_SIZE = 10000;


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


def sds_algorithm(segments):
    """
    apply the sds algorithm on each segment from all the trajectories
    """
    result = []
    stay_num, travel_num = 0, 0

    for seg in segments:
        # the segment with less than three records can not be labeled by our algorithm
        if len(seg) < 3:
            result.append(seg)
            continue

        # label STAY trips in the segment
        # the algorithm below refers to the Algorithm 2 in the paper in ShareLatex
        head = 0
        for cursor in xrange(1, len(seg)):
            
            # too-long stay trip, cut here
            if ((cursor - head) > MAX_STAY_TRIP_SIZE):
                print ('Cut too-long stay trip at segment offset: %d'%(cursor));
                
                if seg[cursor-1][1] - seg[head][1] >= MIN_TIME_INTERVAL:
                    for k in xrange(head, cursor):
                        # only label the record not labeled as stay any more
                        if len(seg[k]) == 4:
                            seg[k].append(0);
                            stay_num = stay_num + 1;
                
                head = cursor;
                continue;
            
            for anchor in xrange(cursor - 1, head - 1, -1):
                space_interval = distance(
                    seg[cursor][2], seg[cursor][3], seg[anchor][2], seg[anchor][3])

                if space_interval > MAX_SPACE_INTERVAL/2:
                    if seg[cursor-1][1] - seg[head][1] >= MIN_TIME_INTERVAL:
                        for k in xrange(head, cursor):
                            # only label the record not labeled as stay
                            if len(seg[k]) == 4:
                                seg[k].append(0)
                                stay_num += 1

                    head = anchor + 1
                    break

        # handle the remaining records in the segment
        if seg[len(seg)-1][1] - seg[head][1] >= MIN_TIME_INTERVAL:
            for k in xrange(head, len(seg)):
                # only label the record not labeled as stay any more
                if len(seg[k]) == 4:
                    seg[k].append(0)
                    stay_num += 1

        # label TRAVEL records in the segment
        # the algorithm below refers to the Algorithm 2 in the paper in ShareLatex
        for cursor in xrange(1, len(seg) - 1):
            # for all the unlabeled records till now
            if len(seg[cursor]) == 4:
                left, right = -1, -1

                # find the first out-of-range record on the left of cursor
                for l in reversed(xrange(cursor)):
                    if distance(seg[cursor][2], seg[cursor][3], seg[l][2], seg[l][3]) > MAX_SPACE_INTERVAL:
                        left = l
                        break
                    if seg[cursor][1] - seg[l][1] > MIN_TIME_INTERVAL:
                        break

                # find the first out-of-range record on the right of cursor
                for r in xrange(cursor + 1, len(seg)):
                    if distance(seg[cursor][2], seg[cursor][3], seg[r][2], seg[r][3]) > MAX_SPACE_INTERVAL:
                        right = r
                        break
                    if seg[r][1] - seg[cursor][1] > MIN_TIME_INTERVAL:
                        break

                if right != -1 and left != -1 and seg[right][1] - seg[left][1] <= MIN_TIME_INTERVAL:
                    seg[cursor].append(1)
                    travel_num += 1

        result.append(seg)
    
    return result, stay_num, travel_num


def label_and_compute_sparsity(filename):
    """
    label the file given the filename
    append 0 after the stay record, append 1 after the travel record, do nothing for other records
    """
    start_time = time.time()

    filename_r = input_path + 'P2-' + filename
    filename_w_tjt = output_path + filename + \
        '-trajectory_' + str(minute) + "-" + str(space)
    filename_w_sparsity = output_path + filename + \
        '-sparsity_' + str(minute) + "-" + str(space)
    
    with open(filename_r) as f:
        records = f.readlines()

    c_uid = -1
    segments, tjt = [], []
    labeled_segments, stats = [], []
    
    # divide the records into to segments
    for record in records:
        columns = record.split(',')

        if len(columns) < 4:
            print('An error line in line: ' + str(record))
            continue

        # set record columns
        uid = columns[0]
        time_second = int(columns[1])
        latitude, longtitude = float(columns[2]), float(columns[3])

        # check if it is the same trajectory
        if uid == c_uid:
            tjt.append([uid, time_second, latitude, longtitude])
        else:
            # new uid
            if c_uid != -1:
                # the current uid is valid, segment the trajectory of the current uid (c_uid)
				# sort the trajectory by time
                tjt.sort(key=lambda x: x[1])

                # truncate the trajectory into segments at every time interval larger than Delta_T, stored in segments
		        # the first index of the current segment
                l = 0
                for r in xrange(1, len(tjt)):
                    time_interval = tjt[r][1] -  tjt[r-1][1]
                    if time_interval > MIN_TIME_INTERVAL:
                        segments.append(tjt[l:r])
                        l = r
                
                if l < len(tjt):
                    segments.append(tjt[l:])
                                    
                # label
                result, stay_num, travel_num = sds_algorithm(segments);
                
                # compute global and local sparsity
                global_sparsity, local_sparsity = 0, 0
                local_sparsity_num = 0
                for i in xrange(1, len(tjt)):
                    time_interval = tjt[i][1] - tjt[i-1][1]
                    global_sparsity += time_interval
                    if time_interval < MIN_TIME_INTERVAL:
                        local_sparsity += time_interval
                        local_sparsity_num += 1
                global_sparsity = global_sparsity / (len(tjt) - 1) if len(tjt) > 1 else 0
                local_sparsity = local_sparsity / (local_sparsity_num * MIN_TIME_INTERVAL) if local_sparsity_num > 0 else 0
                
                global_sparsity = format(global_sparsity, '.4f')
                local_sparsity = format(local_sparsity, '.4f')

                # store results
                
                if len(tjt) > 0:
                    labeled_segments.append(result)
                    stats.append([c_uid, global_sparsity, local_sparsity, stay_num, travel_num, len(tjt)])

                # reset
                segments, tjt = [], []

            
            # refresh the arrays to only store the first record of the new trajectory (uid)
            tjt.append([uid, time_second, latitude, longtitude])
            c_uid = uid
    
    # output to file
    with open(filename_w_tjt, 'w') as f:
        for segments in labeled_segments:
            if write_mode == 0:
                for seg in segments:
                    seg = [','.join([str(x) for x in record]) for record in seg]
                    for record in seg:
                        f.write(record + '\n')
            if write_mode == 1:
                segments = [','.join([str(x) for x in record]) for seg in segments for record in seg]
                f.write('|'.join(segments) + '\n')
    with open(filename_w_sparsity, 'w') as f:
        for stat in stats:
            f.write(','.join([str(x) for x in stat]) + '\n')
    
    stay_num, travel_num, all_num = sum([x[3] for x in stats]), sum([x[4] for x in stats]), sum([x[5] for x in stats])
    print('[file %s] time %f, records num %d, stay num %d (%f%%), travel num %d (%f%%)'
        %(filename, time.time() - start_time, all_num, stay_num, stay_num / all_num * 100, travel_num, travel_num / all_num * 100))

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=15)
    pool.map(label_and_compute_sparsity, filelist)
#     for filename in filelist:
#         label_and_compute_sparsity(filename);
