import numpy as np

#path_length = [int(line.rstrip('\n').split(',')[1]) for line in open('path_length_cleaned.txt')]
#print('max: %d, min: %d' %(max(path_length), min(path_length)))
#print('[truncated] max: %d, len: %d' %(max(path_length), len(path_length)))
#np.savetxt('truncated_path_length.txt', path_length, fmt='%d')
#hist_path_length, _ = np.histogram(path_length, bins=100000)

#output = open('results_stats.txt', 'w')
#output.write(','.join([str(h) for h in hist_path_length]))
#output.write('\n')
#output.close()

path = [[line, int(line.rstrip('\n').split(',')[1])] for line in open('stats/path_length_cleaned.txt')]
#d = {}
#for p in path:
#  if p[1] in d:
#    d[p[1]] += 1
#  else:
#    d[p[1]] = 1
#k = [i for i in d.keys() if d[i]>1]
#print(max(k))
#output = open('tr.txt', 'w')
#for i in range(1, max(k)+1):
#  if i in k:
#    output.write(str(d[i])+'\n')
#  else:
#    output.write('0\n')
#output.close()

sum = 0
for p in path:
  sum += p[1]
print(sum)

#output = open('truncated_stats.txt', 'w')
#for p in path:
#  if d[p[1]] > 1:
#    output.write(p[0])
#output.close()

