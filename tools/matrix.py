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

  max_val = np.max(res)
  res /= max_val

  plt.matshow(res, cmap='plasma')
  plt.show()

with open(sys.argv[1], 'rb') as f:
  show(f)
