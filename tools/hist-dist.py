import json
import numpy as np
import matplotlib.pyplot as plt
import sys

category = 'train' if len(sys.argv) < 3 else sys.argv[2]

with open(sys.argv[1]) as f:
  data = json.load(f)
  data = data[category]

  positive = np.array(data['positives'])
  negative = np.array(data['negatives'])

  # the histogram of the data
  plt.hist(positive, 1000, color='green', density=True, alpha=0.5)
  plt.hist(negative, 1000, color='red', density=True, alpha=0.5)

  plt.title(category)
  plt.xlabel('Distance')
  plt.ylabel('Percentage')
  plt.grid(True)
  plt.show()
