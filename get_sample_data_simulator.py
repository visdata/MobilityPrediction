import sys
import numpy as np

input_dir = '/datahouse/yurl/TalkingData/data/P3-SS/'
output_dir = '/datahouse/yurl/TalkingData/data/P3-SS-sample/'

filelist = ['part-'+format(n, '05d') for n in range(4000)]

minute = int(sys.argv[1])
space = int(sys.argv[2])

BIN_NUM = 10
INIT_BIN_NUM = 100
RESAMPLE_NUM = 10000

stats = []
trajectory = []
for filename in filelist:
  print(filename)
  part_trajectory = [x.rstrip() for x in open(input_dir + filename + '-trajectory_' + str(minute) + '-' + str(space))]
  trajectory.append(part_trajectory)

  part_stats = [x.rstrip().split(',') for x in open(input_dir + filename + '-sparsity_' + str(minute) + '-' + str(space))]
  part_stats = [[x[0], float(x[1]), float(x[2]), int(x[3]), int(x[4]), int(x[5])] for x in part_stats]
  stats.append(part_stats)

stats = [x for y in stats for x in y]
trajectory = [x for y in trajectory for x in y]

# Target - 1: global sparsity, 2: local sparsity, 3: disparity
# Bin_index - 1 to 10
for target in [2]:
#   if target == 3:
#     data = [(i, stats[i][3]) for i in range(len(stats)) if abs(stats[i][2]) > 1e-5]
#   elif target == 4:
#     data = [(i, 1 - stats[i][4] / stats[i][8]) for i in range(len(stats))]
#   else:
#     data = [(i, stats[i][target]) for i in range(len(stats))]
  
  # values = [x[1] for x in data]
  values = [x[target] for x in stats]
  
  hist, edges = np.histogram(values, bins=INIT_BIN_NUM)
  sample_index = BIN_NUM
  index = INIT_BIN_NUM
  resample_stat = [(-1, -1, -1) for _ in range(BIN_NUM)]
  remain = INIT_BIN_NUM
  while sample_index:
    if sample_index == 1:
      resample_stat[0] = (0, index)
      break
    sample_bin_num = int(remain / sample_index)
    cur_idx = index - sample_bin_num
    sample_num = sum(hist[cur_idx:index])
    while sample_num < RESAMPLE_NUM and cur_idx > 0:
      cur_idx -= 1
      sample_num += hist[cur_idx]
      sample_bin_num += 1
    resample_stat[sample_index - 1] = (cur_idx, index - 1, sample_num)
    remain = remain - (index - cur_idx)
    sample_index -= 1
    index = cur_idx
  
  print('resample_edges:', resample_stat)

  print(resample_stat[0][0], edges[resample_stat[0][0]])
  bins = [edges[x[0]] for x in resample_stat] + [edges[resample_stat[BIN_NUM-1][1]+1]]
  inds = np.digitize(values, bins)
  print(bins)
  print(len(inds))
  d = [[] for _ in range(BIN_NUM)]
  for bin_index in range(BIN_NUM):
    d[bin_index] = [x for x in range(len(inds)) if inds[x] == bin_index + 1]
    print(len(d[bin_index]))

  with open(output_dir + '-'.join(['stat', str(RESAMPLE_NUM), str(minute), str(space), str(target), str(BIN_NUM)]), 'w') as f:
    f.write("");
          
  for bin_index in range(1, BIN_NUM+1):
    # set replacement to true: sample without replacement
    indices = np.random.choice(d[bin_index-1], RESAMPLE_NUM, False)
    train_set = indices[:RESAMPLE_NUM]

    train_set_stat = [bins[bin_index-1], 0, len(train_set)]
    for i in train_set:
      train_set_stat[1] += stats[i][target]

    train_set_stat[1] = train_set_stat[1]/len(train_set)
    
    print('train set (%d, %d, %d, %d): %f, %f, %d' % (minute, space, target, bin_index, train_set_stat[0], train_set_stat[1], train_set_stat[2]))

    with open(output_dir + '-'.join(['train', str(RESAMPLE_NUM), str(minute), str(space), str(target), str(BIN_NUM), str(bin_index)]), 'w') as f:
      for i in train_set:
        for x in trajectory[i].split('|'):
          f.write(x)
          f.write('\n')
          
    with open(output_dir + '-'.join(['stat', str(RESAMPLE_NUM), str(minute), str(space), str(target), str(BIN_NUM)]), 'a') as f:
      f.write(','.join([str(train_set_stat[n]) for n in range(len(train_set_stat))]));
      f.write('\n')
