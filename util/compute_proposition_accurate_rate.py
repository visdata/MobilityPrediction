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

filelist = ['part-'+format(n, '05d') for n in range(10000) if (n%100==0)]

input_path = '/datahouse/yurl/TalkingData/data/BJ_cleaned_data/'
output_path = '/datahouse/yurl/TalkingData/data/P3-SS-BJ/'

parser = argparse.ArgumentParser()
parser.add_argument('--minute', type=int, dest='minute',
                    help='(required) the time threshold, unit: minute, e.g. 15', required=True)
parser.add_argument('--space', type=int, dest='space',
                    help='(required) the space threshold, unit: meter, e.g. 800', required=True)
args = parser.parse_args()

minute = args.minute
space = args.space

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


def sds_algorithm_stay(segment):
    """
    apply the sds algorithm on a segment
    """
    
    seg = [[column for column in record] for record in segment];

    # the segment with less than three records can not be labeled by our algorithm
    if len(seg) < 3:
        return seg;
        

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
            
            head = cursor;
            continue;
        
        for anchor in xrange(cursor - 1, head - 1, -1):
            space_interval = distance(
                seg[cursor][2], seg[cursor][3], seg[anchor][2], seg[anchor][3])

            if space_interval > MAX_SPACE_INTERVAL:
                if seg[cursor-1][1] - seg[head][1] >= MIN_TIME_INTERVAL:
                    for k in xrange(head, cursor):
                        # only label the record not labeled as stay
                        if len(seg[k]) == 4:
                            seg[k].append(0)

                head = anchor + 1
                break

    # handle the remaining records in the segment
    if seg[len(seg)-1][1] - seg[head][1] >= MIN_TIME_INTERVAL:
        for k in xrange(head, len(seg)):
            # only label the record not labeled as stay any more
            if len(seg[k]) == 4:
                seg[k].append(0)
    
    return seg

def label_and_compute_proposition_cases(filename):
    """
    label the file given the filename
    append 0 after the stay record, append 1 after the travel record, do nothing for other records
    """
    start_time = time.time()
    
    pos_proposition_num, neg_proposition_num = 0, 0;

    filename_r = input_path + 'P2-' + filename
    filename_w_proposition = output_path + filename + \
        '-proposition-count_' + str(minute) + "-" + str(space)
    
    with open(filename_r) as f:
        records = f.readlines()

    c_uid = -1
    segments, tjt = [], []

    # divide the records into to segments
    for record in records:
        columns = record.split(',')

        if len(columns) < 4:
            print('An error line in line: ' + str(record))
            continue

        # set record columns
        uid = columns[0]
        time_second = int(columns[1][0:10])
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
                    
                # reset
                tjt = []

            # refresh the arrays to only store the first record of the new trajectory (uid)
            tjt.append([uid, time_second, latitude, longtitude])
            c_uid = uid
    
    # print('Finish segment records, %d segments'%(len(segments)))
    
    # process each segment
    for segment_index in range(len(segments)):
        
#         if (segment_index % 10000 == 0):
#             print('Process %d segments in time %f'%(segment_index, time.time() - start_time))
        
        segment = segments[segment_index]
        
        segment_length = len(segment)
        
        # check each record of the segment except the first and last record
        for check_index in range(1,segment_length-1):
            
            check_time = segment[check_index][1]
            check_lat = segment[check_index][2]
            check_lon = segment[check_index][3]
            
            # remove the checked record from the segment and only leave the segment within Delta T
            segment_to_label = [segment[i] for i in range(segment_length) if i!=check_index and abs(segment[i][1]-check_time)<=MIN_TIME_INTERVAL]
            
            left_index_of_check = -1
            
            for i in range(len(segment_to_label)):
                if check_time >= segment_to_label[i][1]:
                    left_index_of_check = i;
                else:
                    break;
            
            if (len(segment_to_label)< 2) or (left_index_of_check<0) or (left_index_of_check>=(len(segment_to_label)-1)):
                continue;
            
            # label and generate the new labeled segment (do not change the previous segment
            segment_labeled = sds_algorithm_stay(segment_to_label)
            
            # the checked record not in the stay segment
            if (len(segment_labeled[left_index_of_check])<5) or (len(segment_labeled[left_index_of_check+1])<5):
                continue;
            
            # shrink the labeled segment to the stay segment containing the checked record
            left_boundary = left_index_of_check;
            for i in range(left_index_of_check-1,-1,-1):
                if (len(segment_labeled[i])>=5):
                    left_boundary = i;
                else:
                    break;
            
            right_boundary = left_index_of_check+1;
            for i in range(left_index_of_check+2,len(segment_labeled)):
                if (len(segment_labeled[i])>=5):
                    right_boundary = i;
                else:
                    break;
            
            segment_labeled = segment_labeled[left_boundary:right_boundary+1];
            left_index_of_check = left_index_of_check - left_boundary;
            
            # compute the distance from checked record to each record in the segment_labeled
            segment_dist_array = [(distance(check_lat, check_lon, segment_labeled[i][2], segment_labeled[i][3])< MAX_SPACE_INTERVAL) for i in range(len(segment_labeled))];
        
            for i in range(left_index_of_check+1):
                for j in range(left_index_of_check+1, len(segment_labeled)):
                    
                    if abs(segment_labeled[i][1]-segment_labeled[j][1])>=MIN_TIME_INTERVAL:
                        break;
                    
                    # satisfy the proposition
                    if segment_dist_array[i] and segment_dist_array[j]:
                        pos_proposition_num = pos_proposition_num + 1;
                    # disobey the proposition
                    else:
                        neg_proposition_num = neg_proposition_num + 1;
    
    
    all_num = pos_proposition_num + neg_proposition_num;
    
    # output to file    
    with open(filename_w_proposition, 'w') as f:
        f.write(','.join([str(pos_proposition_num), str(neg_proposition_num)]) + '\n')
    
    print('[file %s] time %f, positive num %d (%f%%), negative num %d (%f%%)'
        %(filename, time.time() - start_time, pos_proposition_num, float(pos_proposition_num)/all_num, neg_proposition_num, float(neg_proposition_num)/all_num))


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=10)
    pool.map(label_and_compute_proposition_cases, filelist)
#     for filename in filelist:
#         label_and_compute_proposition_cases(filename)
