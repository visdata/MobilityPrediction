import math
import time
import multiprocessing

MAX_SPACE = 800
MAX_TIME = 15 * 60

# Calculate distance between latitude longitude pairs
def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c * 1000 # return meter

    return d

def count_triangle(arr):
  tri1, tri2 = 0, 0
  n = len(arr)
  d = [[False for y in range(n)] for x in range(n)]
  right_bound = [-1 for _ in range(n)]
  for i in range(n-1):
    for j in range(i+1, n):
      if distance((arr[i][2], arr[i][3]), (arr[j][2], arr[j][3])) < MAX_SPACE:
        d[i][j] = d[j][i] = True
      if arr[j][1] - arr[i][1] < MAX_TIME:
        right_bound[i] = j
  for i in range(n):
    for j in range(i+1, right_bound[i]):
      if d[i][j] and d[j][right_bound[i]]:
        tri1 += 1
      tri2 += 1

  return tri1, tri2

def get_tri(filename):
  start_time = time.time()

  tri1, tri2 = 0, 0
  records = [x.rstrip() for x in open(
      '/datahouse/yurl/TalkingData/data/BJ_cleaned_data/P2-' + filename, encoding='utf-8')]
  trajectory = []
  uid, idx = '-1', -1
  for r in records:
    r = r.split(',')
    r = [r[0], int(r[1]), float(r[2]), float(r[3])]
    if r[0] != uid:
      idx += 1
      uid = r[0]
      trajectory.append([r])
    else:
      trajectory[idx].append(r)
  
  for i in range(len(trajectory)):
    trajectory[i] = sorted(trajectory[i], key=lambda tjt: int(tjt[1]))
    tri1_tjt, tri2_tjt = count_triangle(trajectory[i])
    tri1 += tri1_tjt
    tri2 += tri2_tjt

  with open('temp/' + filename, 'w') as f:
    f.write('%s,%d,%d,%f' % (filename, tri1, tri2, tri1 / tri2))

  print('time for file %s: %f' %(filename, time.time() - start_time))


filelist = ['part-' + format(n, '05d') for n in range(4000)]
pool = multiprocessing.Pool(processes=15)
pool.map(get_tri, filelist)
