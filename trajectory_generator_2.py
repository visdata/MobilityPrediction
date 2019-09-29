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
import random
import multiprocessing

input_path = '/datahouse/yurl/TalkingData/data/P3-SS-BJ-inputdata/'
output_path = '/datahouse/yurl/TalkingData/data/BJ-simulation-inputdata/'

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
avg_speed_kmh = args.speed;
avg_speed = args.speed * 3.6

travel_prob = (float(6390)/avg_speed)/((float(6390)/avg_speed) + 7116.4)

# write_mode = args.write_mode

# radius of the earth by km
RADIUS_EARTH = 6371
DEGREE_TO_RADIAN = 2 * math.pi / 360
COS_LATITUDE = 0.77

MAX_SPACE_INTERVAL = space
MIN_TIME_INTERVAL = minute * 60
SPLIT = 0.001

MAX_STAY_TRIP_SIZE = 10000;

RANDOM_ID_OFFSET = 1234567;

filelist = ['train-100000-30-800']

def convert_to_hour(seconds):
    hour = int((seconds - 1467000000) / 3600) % (7 * 24)
    return hour

def convert_longitude(data, split):
    return int((data - 115.422) / split)

def convert_latitude(data, split):
    return int((data - 39.445) / split)

def convert_longitude_to_meter(data):
    return data * DEGREE_TO_RADIAN * RADIUS_EARTH * COS_LATITUDE * 1000;

def convert_latitude_to_meter(data):
    return data * DEGREE_TO_RADIAN * RADIUS_EARTH * 1000;

def convert_meter_to_longitude(data):
    return data / float(DEGREE_TO_RADIAN * RADIUS_EARTH * COS_LATITUDE * 1000);

def convert_meter_to_latitude(data):
    return data / float(DEGREE_TO_RADIAN * RADIUS_EARTH * 1000);

def convert_to_true_second(seconds):
    return int(seconds + 1467000000)

def convert_to_true_longitude(data):
    return data + 150;

def convert_to_true_latitude(data):
    return data + 60;

def distance(lat1, lon1, lat2, lon2):
    """
    compute distance given two points
    """

    lat1 = lat1 * DEGREE_TO_RADIAN
    lon1 = lon1 * DEGREE_TO_RADIAN
    lat2 = lat2 * DEGREE_TO_RADIAN
    lon2 = lon2 * DEGREE_TO_RADIAN
    x = (lon2 - lon1) * COS_LATITUDE
    y = lat2 - lat1
    return int(RADIUS_EARTH * sqrt(x * x + y * y) * 1000)

def generate_new_trajectory_data(filename):
    """
    read the original trajectories from the filename
    generate new trajectories using the generator
    extract the sampling time series of each trajectory from the original data set
    sample in the generated new trajectory, append 0/1 after the stay/travel records (true labels for every record)
    output the new trajectories with true labels into the output file
    """
    start_prog_time = time.time()

    filename_r = input_path + filename
    filename_w_sim = output_path + 'SIM-' +  filename + '-trajectory_' + str(avg_speed_kmh)
    
    with open(filename_r) as f:
        records = f.readlines()

    # refresh the write file to empty
    with open(filename_w_sim, 'w') as f:
        f.write("");
        
    c_uid = -1
    trajectories, tjt = [], []

    # divide the records into to trajectories
    for record in records:
        columns = record.split(',')

        if len(columns) < 4:
            print('An error line in line: ' + str(record))
            continue

        # set record columns
        uid = columns[0]
        time_second = int(columns[1][0:10])

        # check if it is the same trajectory
        if uid == c_uid:
            tjt.append([uid, time_second])
        else:
            # new uid
            if c_uid != -1:
                # the current uid is valid, sort the trajectory of the current uid (c_uid)
				# store the trajectory
                tjt.sort(key=lambda x: x[1])
                trajectories.append(tjt)
      
            # refresh the arrays to only store the first record of the new trajectory (uid)
            tjt = []
            tjt.append([uid, time_second])
            c_uid = uid
    
    stay_num, travel_num = 0, 0;
    
    # generate simulated trajectory for each trajectory read from the file
    for trajectory in trajectories:
        # time span of the trajectory
        trajectory_time_span = trajectory[-1][1] - trajectory[0][1];
        
        # generate the trajectory of the given time span
        # trajectory = [segments]; segment = [start_time, end_time, start_lat, start lon, end_lat, end_lon, angle, speed, stay/travel state]
        
        sim_trajectory = []
        
        start_time, start_lon, start_lat = 0, 0, 0
        end_time, end_lon, end_lat = 0, 0, 0
        trajectory_state = 0
        
        if random.random() < travel_prob:
            trajectory_state = 1
        
        random_time_offset = int(random.random() * 7 * 24 * 60 * 60)
        random_lon_offset = float(random.random() * 10 - 5)
        random_lat_offset = float(random.random() * 10 - 5)
        
        # generate segments until the time span
        while start_time <= trajectory_time_span:
            
            if trajectory_state == 0:
                # stay segment
                stay_time = 3600/(0.04166667 + 1.95833333 * random.random())
                end_time = start_time + stay_time
                end_lon = start_lon
                end_lat = start_lat
                angle = 0
                speed = 0
                
            else:
                # travel segment                                
                angle = random.random() * 2 * math.pi;            
                travel_dist = 1000/(0.004166667 + 0.82916667 * random.random());
                speed = random.uniform(0.5 * avg_speed, 1.5 * avg_speed);
                travel_time = travel_dist / float(speed);
                end_time = start_time + travel_time;
                dist_lon = travel_dist * math.cos(angle);
                dist_lat = travel_dist * math.sin(angle);
                dist_lon = convert_meter_to_longitude(dist_lon);
                dist_lat = convert_meter_to_latitude(dist_lat);
                end_lon = start_lon + dist_lon;
                end_lat = start_lat + dist_lat;
            
            # store the current segment
            sim_trajectory.append([start_time, end_time, start_lat, start_lon, end_lat, end_lon, angle, speed, trajectory_state])
            
            # rotate the stay/travel state
            trajectory_state = 1 - trajectory_state;
            start_time = end_time;
            start_lon = end_lon;
            start_lat = end_lat;
            
        # generate the sampled data set according to the new trajectory and the old time series    
        new_sampled_trajectory = [];
        trajectory_start_time = trajectory[0][1];
        cur_sim_segment = 0;
        
        for record in trajectory:
            
            record_time = record[1] - trajectory_start_time;
            
            # find the segment in the simulated trajectory
            
            while record_time > sim_trajectory[cur_sim_segment][1]:
                cur_sim_segment = cur_sim_segment + 1;
                            
            record_state = sim_trajectory[cur_sim_segment][8];
            
            if record_state == 0:
                # stay segment
                record_label = 0;
                random_angle = random.random() * 2 * math.pi;    
                random_dist_offset = random.random() * float(MAX_SPACE_INTERVAL) / 4
                offset_lat = convert_meter_to_latitude(random_dist_offset * math.sin(random_angle));
                offset_lon = convert_meter_to_longitude(random_dist_offset * math.cos(random_angle));

                record_lat = sim_trajectory[cur_sim_segment][2] + offset_lat;
                record_lon = sim_trajectory[cur_sim_segment][3] + offset_lon;
                stay_num = stay_num + 1;
            else:
                # travel segment
                record_angle = sim_trajectory[cur_sim_segment][6];
                record_speed = sim_trajectory[cur_sim_segment][7];
                record_start_time = sim_trajectory[cur_sim_segment][0];
                record_end_time = sim_trajectory[cur_sim_segment][1];
                
                record_dist = record_speed * (record_time - record_start_time);
                record_dist_to_stop = record_speed * (record_end_time - record_time);
                
                # stay/travel by the definition
                if (record_dist > float(MAX_SPACE_INTERVAL)/2) and (record_dist_to_stop > float(MAX_SPACE_INTERVAL)/2):
                    record_label = 1;
                    travel_num = travel_num + 1;
                else:
                    record_label = 0;
                    stay_num = stay_num + 1;
                
                record_lat = convert_meter_to_latitude(record_dist * math.sin(record_angle)) + sim_trajectory[cur_sim_segment][2];
                record_lon = convert_meter_to_longitude(record_dist * math.cos(record_angle)) + sim_trajectory[cur_sim_segment][3];
            
            new_sampled_trajectory.append([int(record[0])+RANDOM_ID_OFFSET,int(float(record[1])+random_time_offset), convert_to_true_latitude(record_lat) + random_lat_offset, convert_to_true_longitude(record_lon) + random_lon_offset, record_label]);
        
    
        # output to file
        with open(filename_w_sim, 'a') as f:
            for record in new_sampled_trajectory:
                record_str = ','.join([str(x) for x in record])
                f.write(record_str + '\n')
    
#         print('Output one trajectory!');
    
    all_num = stay_num + travel_num;            
    print('[file %s] time %f, records num %d, stay num %d (%f%%), travel num %d (%f%%)'
        %(filename, time.time() - start_prog_time, all_num, stay_num, float(stay_num) / all_num * 100, travel_num, float(travel_num) / all_num * 100))


if __name__ == "__main__":
#     pool = multiprocessing.Pool(processes=15)
#     pool.map(generate_new_trajectory_data, filelist)
    for filename in filelist:
        generate_new_trajectory_data(filename);
