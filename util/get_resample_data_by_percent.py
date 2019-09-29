import numpy as np

input_dir = '/datahouse/yurl/TalkingData/P4-resample/'
output_dir = '/datahouse/yurl/TalkingData/P4-resample/'

filename = 'test-10000-15-800-2-100-13'

def sample(arr, percent, mode='random'):
  import math
  n = int(math.ceil(len(arr) * percent))
  if n == 1:
    return [arr[0]]
  else:
    gap = min(len(arr) - 1, len(arr) // (n - 1))
  
  if mode == 'random':
    inds = [i for i in range(len(arr))]
    np.random.seed(int(2018 * percent))
    inds = np.random.choice(inds, n, replace=False)
    inds.sort()
    new_arr = [arr[i] for i in inds]
  
  if mode == 'uniform':
    start = 0
    new_arr = []
    while n > 0:
      n -= 1
      new_arr.append(start)
      start += gap

  return new_arr
  

trajectory = [x.rstrip() for x in open(input_dir + filename)]
for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
  new_trajectory = [sample(x.split('|'), p, 'random') for x in trajectory]
  with open(output_dir + filename + '-' + str(p), 'w') as f:
    for x in new_trajectory:
      for t in x:
        f.write(t)
        f.write('\n')

