import sys
import numpy as np

basepath = '/datahouse/yurl/TalkingData/data/P3-SS-BJ/'

filelist = ['part-'+format(n, '05d') for n in range(10000)]

minute = int(sys.argv[1])
space = int(sys.argv[2])
sparsity_choice = int(sys.argv[3])

BIN_NUM = 1000
if(len(sys.argv)>4):
   BIN_NUM = int(sys.argv[4])

stats = []
cnt = 0
for filename in filelist:
  cnt += 1
  if cnt % 200 == 0:
    print(filename)
  part_stats = [x.rstrip().split(',') for x in open(basepath + filename + '-sparsity_' + str(minute) + '-' + str(space))]
  part_stats = [[x[0], float(x[1]), float(x[2]), int(x[3]), int(x[4]), int(x[5])] for x in part_stats]
  stats.append(part_stats)

stats = [x for y in stats for x in y]
values = [[] for _ in range(5)]
hist = [[] for _ in range(5)]
edges = [[] for _ in range(5)]
travel_num = [[] for _ in range(5)]
stay_num = [[] for _ in range(5)]
all_num = [[] for _ in range(5)]
avg_x = [[] for _ in range(5)]

# global/local_sparsities
values[sparsity_choice] = [x[sparsity_choice] for x in stats]
# sorted_values = [x[sparsity_choice] for x in stats]

# sorted_values.sort()

# bin_width_by_value = float(len(stats))/ BIN_NUM

#bin_edges = [sorted_values[int(bin_width_by_value*i)] for i in range(BIN_NUM)]
#bin_edges.append(sorted_values[len(stats)-1])

hist[sparsity_choice], edges[sparsity_choice] = np.histogram(values[sparsity_choice], bins=BIN_NUM)

stay = [x[3] for x in stats]
travel = [x[4] for x in stats]
all_records = [x[5] for x in stats]


target = sparsity_choice;

travel_num[target] = [0] * BIN_NUM
stay_num[target] = [0] * BIN_NUM
all_num[target] = [0] * BIN_NUM
avg_x[target] = [0] * BIN_NUM

inds = np.digitize(values[target], edges[target])

for i in range(len(stats)):
  x = stats[i]
  bin_index = min(inds[i], BIN_NUM) - 1
  value = x[target]
  
  stay_num[target][bin_index] += x[3]
  travel_num[target][bin_index] += x[4]
  all_num[target][bin_index] += x[5]

  avg_x[target][bin_index] += values[target][i]

avg_x[target] = [avg_x[target][i] / hist[target][i] if hist[target][i] > 0 else (edges[target][i] + edges[target][i+1]) / 2 for i in range(BIN_NUM)]

def write_array_to_file(f, arr, sep=','):
  f.write(sep.join([str(x) for x in arr]))
  f.write('\n')

with open(basepath+'/overall_stats/stats_' + str(minute) + '-' + str(space) + '-' + str(sparsity_choice) + '-' + str(BIN_NUM) +'.csv', 'w') as f:
  for i in range(BIN_NUM):
      if all_num[sparsity_choice][i]>0:
          if sparsity_choice==1:
              array_to_file = [float(avg_x[sparsity_choice][i])/60, float(stay_num[sparsity_choice][i])/all_num[sparsity_choice][i], float(travel_num[sparsity_choice][i])/all_num[sparsity_choice][i], edges[sparsity_choice][i], stay_num[sparsity_choice][i], travel_num[sparsity_choice][i], all_num[sparsity_choice][i], hist[sparsity_choice][i]];
          else:
              array_to_file = [avg_x[sparsity_choice][i], float(stay_num[sparsity_choice][i])/all_num[sparsity_choice][i], float(travel_num[sparsity_choice][i])/all_num[sparsity_choice][i], edges[sparsity_choice][i], stay_num[sparsity_choice][i], travel_num[sparsity_choice][i], all_num[sparsity_choice][i], hist[sparsity_choice][i]];
      else:
          continue;
#           if sparsity_choice==1:
#               array_to_file = [float(avg_x[sparsity_choice][i])/60, 0, 0, edges[sparsity_choice][i], stay_num[sparsity_choice][i], travel_num[sparsity_choice][i], all_num[sparsity_choice][i], hist[sparsity_choice][i]];
#           else:
#               array_to_file = [avg_x[sparsity_choice][i], 0, 0, edges[sparsity_choice][i], stay_num[sparsity_choice][i], travel_num[sparsity_choice][i], all_num[sparsity_choice][i], hist[sparsity_choice][i]];
      
      write_array_to_file(f, array_to_file)

if sparsity_choice==1:
    print('global sparsity: max %f, min %f, average %f' %(max(values[1]), min(values[1]), sum(values[1]) / len(values[1])))

if sparsity_choice==2:
    print('local sparsity: max %f, min %f, average %f' %(max(values[2]), min(values[2]), sum(values[2]) / len(values[2])))

print('stay: %d (%.2f), travel: %d (%.2f), all: %d' % (sum(stay), 100*float(sum(stay))/sum(all_records), sum(travel), 100*float(sum(travel))/sum(all_records), sum(all_records)));

# for i in range(len(disparities)):
#   if disparities[i] > 22810000:
#     print(global_sparsities[i], local_sparsities[i], disparities[i])
