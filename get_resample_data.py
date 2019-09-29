import sys
import numpy as np

input_dir = '/datahouse/yurl/TalkingData/P4/'
output_dir = '/datahouse/yurl/TalkingData/P4-resample/'

filelist = ['part-'+format(n, '05d') for n in range(4000)]

minute = int(sys.argv[1])
space = int(sys.argv[2])

BIN_NUM = 100
RESAMPLE_NUM = 10000

stats = []
trajectory = []
for filename in filelist:
  print(filename)
  part_trajectory = [x.rstrip() for x in open(input_dir + filename + '-trajectory_' + str(minute) + '-' + str(space))]
  trajectory.append(part_trajectory)

  part_stats = [x.rstrip().split(',') for x in open(input_dir + filename + '-sparsity_' + str(minute) + '-' + str(space))]
  part_stats = [[x[0], float(x[1]), float(x[2]), float(x[3]), int(x[4]), int(x[5]), int(x[6]), int(x[7]), int(x[8])] for x in part_stats]
  stats.append(part_stats)

stats = [x for y in stats for x in y]
trajectory = [x for y in trajectory for x in y]

# Target - 1: global sparsity, 2: local sparsity, 3: disparity
# Bin_index - 1 to 10
for target in [2]:
  values = [x[target] for x in stats]
  travel_num = [0] * BIN_NUM
  stay_num = [0] * BIN_NUM
  all_num = [0] * BIN_NUM
  
  hist, edges = np.histogram(values, bins=BIN_NUM)
  inds = np.digitize(values, edges)
  for i in range(len(inds)):
    ind = min(inds[i], BIN_NUM) - 1
    # if ind >= BIN_NUM:
    #   print(inds[i], ind, values[i], edges[ind])
    travel_num[ind] += stats[i][6]
    stay_num[ind] += stats[i][7]
    all_num[ind] += stats[i][8]
  ratio = [(travel_num[i] + stay_num[i]) / all_num[i] for i in range(BIN_NUM)]
  max_ind = ratio.index(max(ratio))
  d = [i for i in range(len(inds)) if inds[i] == max_ind + 1]

  print(max_ind)
  print(ratio)
  
  # set replacement to true: sample without replacement
  np.random.seed(2018)
  indices = np.random.choice(d, 4 * RESAMPLE_NUM, False)
  train_set = indices[:2*RESAMPLE_NUM]
  test_set = indices[2*RESAMPLE_NUM:4*RESAMPLE_NUM]

  train_tjt_num = 0
  with open(output_dir + '-'.join(['train', str(RESAMPLE_NUM), str(minute), str(space), str(target), str(BIN_NUM), str(max_ind)]), 'w') as f:
    for i in train_set:
      if train_tjt_num == RESAMPLE_NUM:
        break
      valid = [x for x in trajectory[i].split('|') if len(x.split(',')) == 5]
      if len(valid) > 0:
        f.write('|'.join(valid))
        f.write('\n')
        train_tjt_num += 1
      # for x in trajectory[i].split('|'):
      #   if len(x.split(',')) == 6:
      #     f.write(x)
      #     f.write('\n')

  test_tjt_num = 0
  with open(output_dir + '-'.join(['test', str(RESAMPLE_NUM), str(minute), str(space), str(target), str(BIN_NUM), str(max_ind)]), 'w') as f:
    for i in test_set:
      if test_tjt_num == RESAMPLE_NUM:
        break
      valid = [x for x in trajectory[i].split('|') if len(x.split(',')) == 5]
      if len(valid) > 0:
        f.write('|'.join(valid))
        f.write('\n')
        test_tjt_num += 1
      # for x in trajectory[i].split('|'):
      #   if len(x.split(',')) == 6:
      #     f.write(x)
      #     f.write('\n')
  print(train_tjt_num, test_tjt_num)
