import sys
import numpy as np
import time

input_dir = '/datahouse/yurl/TalkingData/data/P3-SS-BJ/'
output_dir = '/datahouse/yurl/TalkingData/data/P3-SS-BJ-inputdata/'

filelist = ['part-'+format(n, '05d') for n in range(5000)]

minute = int(sys.argv[1])
space = int(sys.argv[2])

start_time = time.time()

MIN_TRAVEL_NUM = 10
MIN_TRAVEL_RATIO = 0.01

stats = []
trajectory = []
filenum = 0;
for filename in filelist:
	if filenum%100 == 0:
		print('Process %d file in %f'%(filenum, time.time()-start_time));
	
	part_trajectory = [x.rstrip() for x in open(input_dir + filename + '-trajectory_' + str(minute) + '-' + str(space)) if len(x.strip())>0]
	trajectory.append(part_trajectory)

	part_stats = [x.rstrip().split(',') for x in open(input_dir + filename + '-sparsity_' + str(minute) + '-' + str(space))]
	part_stats = [[x[0], float(x[1]), float(x[2]), int(x[3]), int(x[4]), int(x[5])] for x in part_stats if int(x[5]) > 0]
	stats.append(part_stats)
	filenum = filenum+1

stats = [x for y in stats for x in y]
trajectory = [x for y in trajectory for x in y]

large_travel_trajectory_set = []

sample_set_stat = [0, 0, 0, 0]

for trajectory_index in range(len(trajectory)):
	
	travel_num = stats[trajectory_index][4]
	
	travel_ratio = float(travel_num) / stats[trajectory_index][5]

	if travel_num >= MIN_TRAVEL_NUM and travel_ratio >= MIN_TRAVEL_RATIO:
		
		large_travel_trajectory_set.append(trajectory_index);
		
		sample_set_stat[0] += stats[trajectory_index][3]
		sample_set_stat[1] += stats[trajectory_index][4]
		sample_set_stat[2] += stats[trajectory_index][5]
		sample_set_stat[3] += stats[trajectory_index][2]
		
print('large travel trajectory sample (%d, %d): %d, %d, %d, %d, %f' % (minute, space, sample_set_stat[0], sample_set_stat[1], sample_set_stat[2], len(large_travel_trajectory_set), float(sample_set_stat[3])/len(large_travel_trajectory_set)))

with open(output_dir + '-'.join(['large_travel_sample', str(minute), str(space)]), 'w') as f:
	for i in large_travel_trajectory_set:
		f.write(trajectory[i])
		f.write('\n')
