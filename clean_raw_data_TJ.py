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

filelist = ['part-'+format(n, '05d') for n in range(1000)]

input_path = '/enigma/tao.jiang/datasets/JingJinJi/records/TD-rawdata/Tianjin/'
output_path = '/datahouse/yurl/TalkingData/data/TJ_cleaned_data/'

MIN_TIME_INTERVAL = float(45) * 60

MAX_CHECK_WINDOW_SIZE = 10;

MAX_MOVEMENT_SPEED = 40;

MAX_SMOOTHENED_TIME = 10;


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


def is_from_multiple_apps(segment):
    
    for record_index in range(len(segment)):
        
        for checked_index in range(record_index+1, min(record_index+MAX_CHECK_WINDOW_SIZE+1,len(segment))):
        
            record = segment[record_index];
            checked_record = segment[checked_index];
            
            time_interval = abs(record[1] - checked_record[1])
            
            if time_interval < MAX_SMOOTHENED_TIME:
                time_interval = MAX_SMOOTHENED_TIME
            
            speed = distance(record[2], record[3], checked_record[2], checked_record[3])/float(time_interval)
            
            if speed > MAX_MOVEMENT_SPEED:
                return True
    
    return False

def clean_data(filename):
    """
    label the file given the filename
    append 0 after the stay record, append 1 after the travel record, do nothing for other records
    """
    start_time = time.time()

    filename_r = input_path + filename
    filename_w_tjt = output_path + 'P2-' + filename
    
    with open(filename_r) as f:
        records = f.readlines()

    c_uid = -1
    segments, tjt = [], []
    
    cleaned_segments = []
    all_record_num = len(records)
    removed_record_num = 0

    # divide the records into to segments
    for record in records:
        columns = record.split(',')

        if len(columns) < 4:
            print('An error line in line: ' + str(record))
            continue

        # set record columns
        uid = columns[0]
        time_second = int(int(columns[1])/1000)
        latitude, longtitude = round(float(columns[2]),5), round(float(columns[3]),5)

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
                        
                        segment = tjt[l:r]
                        
                        if is_from_multiple_apps(segment):
                            removed_record_num = removed_record_num + len(segment)
                        else:
                            segments.append(segment)
                        
                        l = r
                
                if l < len(tjt):
                    
                    segment = tjt[l:]
                    
                    if is_from_multiple_apps(segment):
                        removed_record_num = removed_record_num + len(segment)
                    else:
                        segments.append(segment)
                                    
                
                if len(segments) > 0:
                    cleaned_segments.append(segments)

                # reset
                segments, tjt = [], []

            
            # refresh the arrays to only store the first record of the new trajectory (uid)
            tjt.append([uid, time_second, latitude, longtitude])
            c_uid = uid
    
    # output to file
    with open(filename_w_tjt, 'w') as f:
        for segments in cleaned_segments:
            for seg in segments:
                seg = [','.join([str(x) for x in record]) for record in seg]
                for record in seg:
                    f.write(record + '\n')
    
    print('[file %s] time %f, records num %d, removed num %d (%f%%)'
        %(filename, time.time() - start_time, all_record_num, removed_record_num, float(removed_record_num)/all_record_num * 100))

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=10)
    pool.map(clean_data, filelist)
#     for filename in filelist:
#         clean_data(filename);
