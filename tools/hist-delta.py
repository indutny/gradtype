import csv
import numpy as np
import matplotlib.pyplot as plt
import sys

with open(sys.argv[1]) as input:
  reader = csv.reader(input, delimiter=',')
  delta = []
  for row in reader:
    delta.append(float(row[2]) * 1000)
  delta = np.array(delta)

  # the histogram of the data
  plt.hist(delta, 1500, facecolor='g', alpha=0.75)

  plt.axis([ 0, 500, 0, 25 ])
  plt.xlabel('Delta')
  plt.ylabel('Count')
  plt.grid(True)
  plt.show()
