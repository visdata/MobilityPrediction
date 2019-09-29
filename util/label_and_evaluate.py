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

input_path = '/datahouse/yurl/TalkingData/data/BJ-simulation/'
output_path = '/datahouse/yurl/TalkingData/data/BJ-simulation/overall_stats/'

parser = argparse.ArgumentParser()
parser.add_argument('--minute', type=int, dest='minute',
                    help='(required) the time threshold, unit: minute, e.g. 15', required=True)
parser.add_argument('--space', type=int, dest='space',
                    help='(required) the space threshold, unit: meter, e.g. 800', required=True)
parser.add_argument('--speed', type=int, dest='speed',
                    help='(required) the average speed, unit: km/hour, e.g. 20', required=True)
# parser.add_argument('--write_mode', type=int, dest='write_mode', default=1,
#                     help='(optional) the output mode (default 1): 1 - write the records line by line, 1 - write all the records of one uid to one line and split the record with |')
args = parser.parse_args()

minute = args.minute
space = args.space
speed = args.speed

MAX_SPACE_INTERVAL = float(space)
MIN_TIME_INTERVAL = float(minute) * 60
SPLIT = 0.001

MAX_STAY_TRIP_SIZE = 10000;

filelist = ['SIM-train-10000-' + str(minute) + "-" + str(space) + "-" +str(format(float(n)/10, '.1f')) for n in range(1,11)]

filename_w_stats = output_path + 'sds-sparsity-perf_' + str(minute) + "-" + str(space) + "-" + str(speed)+".csv"
        
with open(filename_w_stats,'w') as f:
    f.write("")

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
                        if len(seg[k]) == 5:
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
                            if len(seg[k]) == 5:
                                seg[k].append(0)
                                stay_num += 1

                    head = anchor + 1
                    break

        # handle the remaining records in the segment
        if seg[len(seg)-1][1] - seg[head][1] >= MIN_TIME_INTERVAL:
            for k in xrange(head, len(seg)):
                # only label the record not labeled as stay any more
                if len(seg[k]) == 5:
                    seg[k].append(0)
                    stay_num += 1

        # label TRAVEL records in the segment
        # the algorithm below refers to the Algorithm 2 in the paper in ShareLatex
        for cursor in xrange(1, len(seg) - 1):
            # for all the unlabeled records till now
            if len(seg[cursor]) == 5:
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


def label_and_evaluate(filename):
    """
    label the file given the filename
    append 0 after the stay record, append 1 after the travel record, do nothing for other records
    """
    start_time = time.time()

    filename_r = input_path + filename + '-trajectory_' + str(speed)
    
    with open(filename_r) as f:
        records = f.readlines()

    c_uid = -1
    segments, tjt = [], []
    labeled_segments = []
    total_global_sparsity = 0
    interval_num = 0

    # divide the records into to segments
    for record in records:
        columns = record.split(',')

        if len(columns) < 5:
            print('An error line in line: ' + str(record))
            continue

        # set record columns
        uid = columns[0]
        time_second = int(columns[1])
        latitude, longitude = float(columns[2]), float(columns[3])
        label = int(columns[4])

        # check if it is the same trajectory
        if uid == c_uid:
            tjt.append([uid, time_second, latitude, longitude, label])
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
                
                result, stay_num, travel_num = sds_algorithm(segments)
                
                # compute global sparsity
                global_sparsity = 0
                for i in xrange(1, len(tjt)):
                    time_interval = tjt[i][1] - tjt[i-1][1]
                    global_sparsity += time_interval
                    
                global_sparsity = float(global_sparsity) / 60
                total_global_sparsity = total_global_sparsity + global_sparsity
                interval_num = interval_num + (len(tjt) - 1)

                # store results
                labeled_segments.append(result)

                # reset
                segments, tjt = [], []

            
            # refresh the arrays to only store the first record of the new trajectory (uid)
            tjt = [];
            tjt.append([uid, time_second, latitude, longitude, label])
            c_uid = uid
    
    # compute the perf
    true_stay_num, true_travel_num, false_stay_num, false_travel_num, unlabeled_stay_num, unlabeled_travel_num = 0, 0, 0, 0, 0, 0;
    
    for segments in labeled_segments:
        for seg in segments:
            for record in seg:
                if len(record) == 6:
                    # labeled record
                    if int(record[4])==int(record[5]):
                        if int(record[5]) == 0:
                            true_stay_num = true_stay_num + 1;
                        else:
                            true_travel_num = true_travel_num + 1;
                    else:
                        if int(record[5]) == 0:
                            false_stay_num = false_stay_num + 1;
                        else:
                            false_travel_num = false_travel_num + 1;
                            
                elif len(record) == 5:
                    # unlabeled record
                    if int(record[4]) == 0:
                        unlabeled_stay_num = unlabeled_stay_num + 1;
                    else:
                        unlabeled_travel_num = unlabeled_travel_num + 1;
                else:
                    print("wrong record!");
            
    all_record_num = true_stay_num+true_travel_num+false_stay_num+false_travel_num+unlabeled_stay_num+unlabeled_travel_num;
    org_stay_num = true_stay_num + false_travel_num + unlabeled_stay_num;
    org_travel_num = true_travel_num + false_stay_num + unlabeled_travel_num;
    pred_stay_num = true_stay_num + false_stay_num;
    pred_travel_num = true_travel_num + false_travel_num;
    
    stay_precision = float(true_stay_num)/(true_stay_num + false_stay_num)
    travel_precision = float(true_travel_num)/(true_travel_num + false_travel_num)
    stay_recall = float(true_stay_num)/(true_stay_num + false_travel_num + unlabeled_stay_num)
    travel_recall = float(true_travel_num)/(true_travel_num + false_stay_num + unlabeled_travel_num)
    acc = float(true_stay_num +true_travel_num)/all_record_num
    
    
    if stay_precision + stay_recall == 0:
        stay_f_measure = 0
    else:
        stay_f_measure = 2 * stay_precision * stay_recall /(stay_precision + stay_recall)
    
    if travel_precision + travel_recall == 0:
        travel_f_measure = 0
    else:
        travel_f_measure = 2 * travel_precision * travel_recall/(travel_precision + travel_recall)
    
    if stay_f_measure + travel_f_measure == 0:
        acc_f1 = 0
    else:
        acc_f1 = 2 * stay_f_measure * travel_f_measure/(stay_f_measure + travel_f_measure)
    
    # output to file
    with open(filename_w_stats, 'a') as f:
        write_record = ','.join([filename, str(travel_precision), str(travel_recall), str(stay_precision), str(stay_recall), str(acc), str(travel_f_measure), str(stay_f_measure), str(acc_f1), str(pred_travel_num), str(pred_stay_num), str(org_travel_num), str(org_stay_num), str(all_record_num), str(len(labeled_segments)), str(float(total_global_sparsity)/interval_num)])
        f.write(write_record + '\n')
    
    print('[file %s] time %f, stay_precision %f, travel_precision %f, stay_recall %f, travel_recall %f, stay_f_measure %f, travel_f_measure %f, w_acc %f'
        %(filename, time.time() - start_time, stay_precision, travel_precision, stay_recall, travel_recall, stay_f_measure, travel_f_measure, acc_f1))


if __name__ == "__main__":
#     pool = multiprocessing.Pool(processes=15)
#     pool.map(label_and_evaluate, filelist)
    for filename in filelist:
        label_and_evaluate(filename)
