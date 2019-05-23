import json
import numpy as np
import matplotlib.pyplot as plt
import sys

category = 'train' if len(sys.argv) < 3 else sys.argv[2]

fig = plt.figure(1, figsize=(8, 8))

with open(sys.argv[1]) as f:
  raw = json.load(f)

  if 'step' in raw:
    step = 'step {}'.format(raw['step'])
  else:
    step = 'unknown'

  for category in [ 'train', 'validate' ]:
    data = raw[category]

    positive = np.array(data['positives'])
    negative = np.array(data['negatives'])

    plt.subplot(2, 1, 1 if category is 'train' else 2)

    # the histogram of the data
    plt.hist(positive, 1000, color='green', density=True, alpha=0.5)
    plt.hist(negative, 1000, color='red', density=True, alpha=0.5)

    plt.title(category + ' / ' + step)
    plt.xlabel('Distance')
    plt.ylabel('Percentage')
    plt.grid(True)

  plt.tight_layout()
  plt.show()
