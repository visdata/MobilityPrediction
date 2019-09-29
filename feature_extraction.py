import sys;

"""
the edge of the period
'beijing': {
			'north':  41.055,
			'south':  39.445,
			'west':  115.422,
			'east':  117.515
		}
Time Reference:
1467000000: Mon 00:00:00 2016-6-27
"""

# convert seconds to hour
def convert_to_hour(seconds):
	# hour = int((seconds - 1467000000 - int((seconds - 1467000000) / (24 * 7 * 3600)) * (24 * 7 * 3600)) / 3600)
	hour = int((seconds - 1467000000)) / 3600 % (7*24)
	return hour
# convert longitude/latitude to one-hot
def convert_longitude(data, split):
	return int((data - 115.422) / split)

def convert_latitude(data, split):
	return int((data - 39.445) / split)

SPLIT = 0.001;

input_path = "/datahouse/yurl/TalkingData/data/"
output_path = "/datahouse/yurl/TalkingData/data/"
filename = sys.argv[1]
minute = sys.argv[2]

MIN_TIME_INTERVAL = float(minute) * 60

trajectory = [x.rstrip() for x in open(input_path + filename)]

src_time = -1
last_time = -1
c_uid = -1
new_trajectory = []
for x in trajectory:
	columns = x.split(",")
	uid = columns[0]
	seconds = int(columns[1])
	lat = float(columns[2])
	lon = float(columns[3])
	state = -1
	if len(columns) > 4:
		state = int(columns[4])
	if src_time == -1:
		src_time = seconds
		last_time = seconds
		c_uid = uid
	if seconds - last_time > MIN_TIME_INTERVAL or c_uid != uid:
		src_time = seconds
		last_time = seconds
		c_uid = uid
	last_time = seconds
	minutes = int((seconds - src_time) / 6)
	hours = convert_to_hour(seconds)
	lat = convert_latitude(lat, SPLIT)
	lon = convert_longitude(lon, SPLIT)
	if hours < 0 or minutes >= 10000:
		continue
	line = uid + ',' + str(hours) + ',' + str(minutes) + ',' + str(lat) + ',' + str(lon)
	if len(columns) <= 4:
		new_trajectory.append(line)
	else:
		new_trajectory.append(line + ',' + str(state))

with open(output_path + filename + "_f", 'w') as ofile:
	for x in new_trajectory:
		ofile.write(x)
		ofile.write('\n')

