import csv
import numpy as np
import matplotlib.pyplot as plt
import sys

with open(sys.argv[1]) as input:
  reader = csv.reader(input, delimiter=',')
  delta = []
  for row in reader:
    delta.append(float(row[1]) * 1000)

  # the histogram of the data
  plt.plot(np.arange(0, len(delta), 1), np.array(delta))

  plt.grid(True)
  plt.show()
