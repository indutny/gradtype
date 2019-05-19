import matplotlib
import sys
import json
import struct

import matplotlib.pyplot as plt
import numpy as np

def show(f):
  width = struct.unpack('<i', f.read(4))[0]
  res = np.zeros((width, width,))
  for i in range(0, width):
    for j in range(0, width):
      res[i, j] = struct.unpack('<f', f.read(4))[0]

  res /= 2.0

  plt.matshow(res, cmap='plasma')
  print('matshow done')
  plt.show()
  print('done')

with open(sys.argv[1], 'rb') as f:
  show(f)
