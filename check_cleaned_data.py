import math
import datetime
import multiprocessing

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
    d = radius * c

    return d


filelist = ['P2-part-'+format(n, '05d') for n in range(4000)]
file_path = '/datahouse/yurl/TalkingData/P2/'

# Distance threshold: 100m
dist_thresh_100 = 0.1
dist_thresh_10 = 0.01
dist_thresh_1 = 0.001

def preprocess(filename):
  global counts
  curr_id = -1

  dup_num_100 = 0
  dup_num_10 = 0
  dup_num_1 = 0
  is_dup = False
  is_time_correct = True
  time_err = 0
  id_path_length = 0
  records = {}
  with open(file_path+filename, 'r') as f:
    data = f.readlines()
    num_data = len(data)

    for idx in range(num_data):
      record = data[idx].rstrip('\n').split(',')
      user_id = record[0]
      # Set the unit to second
      time = int(int(record[1]) / 1000)
      latitude = float(record[2])
      longitude = float(record[3])
      district = record[4]

      if user_id != curr_id or idx == num_data-1:
        curr_id = user_id
        records = {}
      
      # Check if time is in year 2016
      year = datetime.datetime.fromtimestamp(time).year
      if (year != 2016):
        print('[file: %s] time error: %s' %(filename, datetime.datetime.fromtimestamp(time).isoformat()))
        continue

      if is_time_correct:
        if time in records:
          for r in records[time]:
            # Existence timestamp: compare
            dist = distance((r[0],r[1]), (latitude,longitude)) 
            if dist < dist_thresh_100:
              print('[file: %s] distance error: %s, %s, (%s, %s), (%s, %s)' %(filename, curr_id, time, r[0], r[1], latitude, longitude))
        else:
          records[time] = [[latitude, longitude, district]] 



pool = multiprocessing.Pool(processes=10)
pool.map(preprocess, filelist)

