import sys
import numpy as np
import random
import time


input_dir = '/datahouse/yurl/TalkingData/data/P3-SS-TS-inputdata/'
output_dir = '/datahouse/yurl/TalkingData/data/P3-SS-TS-inputdata/'

start_time = time.time()

minute = int(sys.argv[1])
space = int(sys.argv[2])

filename = '-'.join(['large_travel_sample', str(minute), str(space)])

sample_data_size = []
sample_data_name = []

sample_data_set_num = len(sys.argv) - 3

for i in range(0, sample_data_set_num):
	sample_data_name.append(sys.argv[i+3])
	sample_data_size.append(int(sys.argv[i+3].strip().split('-')[1]))

total_sample_data_size = sum(sample_data_size)

trajectory = [x.rstrip() for x in open(input_dir + filename)]

if total_sample_data_size > len(trajectory):
	print ('total sample size (%d) is larger than the number of trajectories (%d)'%(total_sample_data_size,len(trajectory)))
	sys.exit()

random.seed()
sample_index_array = [i for i in range(len(trajectory))]
random.shuffle(sample_index_array)

start_index = 0

for sample_data_index in range(len(sample_data_size)):
	
	cur_sample_index_array = sample_index_array[start_index:start_index+sample_data_size[sample_data_index]];
	start_index = start_index + sample_data_size[sample_data_index]
	
	with open(output_dir + '-'.join([sample_data_name[sample_data_index], str(minute), str(space)]), 'w') as f:
		for i in cur_sample_index_array:
			for x in trajectory[i].split('|'):
				f.write(x)
				f.write('\n')
	
