import numpy as np

input_dir = '/datahouse/yurl/TalkingData/P3-resample/'
output_dir = '/datahouse/yurl/TalkingData/P3-resample/'

filename = 'test-10000-15-800-2-100-13'

def sample_encoder_decoder(arr, percent, mode='random'):
  import math
  n = int(math.ceil(len(arr) * percent))
  # if n == 1:
  #   return [arr[0]]
  # else:
  #   gap = min(len(arr) - 1, len(arr) // (n - 1))
  
  if mode == 'random':
    en_inds = [i for i in range(len(arr))]
    np.random.seed(int(2018 * percent))
    if len(en_inds) > 0:
      en_inds = np.random.choice(en_inds, n, replace=False)
      en_inds.sort()

    de_inds = [i for i in range(len(arr)) if i not in en_inds]
    np.random.seed(int(2018 * percent))
    if len(de_inds) > 0:
      de_inds = np.random.choice(de_inds, min(200, len(arr) - n), replace=False)
      de_inds.sort()

    encoder_data = [arr[i] for i in en_inds]
    decoder_data = [arr[i] for i in de_inds]

  return encoder_data, decoder_data
  

trajectory = [x.rstrip() for x in open(input_dir + filename)]
for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
  print(p)
  en_file = open(output_dir + filename + '-' + str(p) + '-' + 'encoder', 'w')
  de_file = open(output_dir + filename + '-' + str(p) + '-' + 'decoder', 'w')

  for x in trajectory:
    encoder, decoder = sample_encoder_decoder(x.split('|'), p, 'random')
    if len(encoder) > 0 and len(decoder) > 0:
      en_file.write('|'.join(encoder))
      en_file.write('\n')
      de_file.write('|'.join(decoder))
      de_file.write('\n')

  en_file.close()
  de_file.close()
