import csv
import numpy as np
import matplotlib.pyplot as plt
import sys

category = 'train' if len(sys.argv) < 3 else sys.argv[2]

NEGATIVE_COLOR = (1.0, 0.098039215686275, 0.223529411764706,)
POSITIVE_COLOR = (0.219607843137255, 0.698039215686275, 0.317647058823529,)

with open(sys.argv[1]) as input:
  reader = csv.reader(input, delimiter=',')
  labels = next(reader)

  steps = []
  data = [ { 'label': label, 'values': [] } for label in labels[2:] ]
  for row in reader:
    steps.append(int(row[1]))
    for value, subdata in zip(row[2:], data):
      subdata['values'].append(float(value))

  for subdata in data:
    label = subdata['label']
    if not category in label:
      continue

    values = np.array(subdata['values'])

    alpha = 1.0 if '_50' in label else \
        0.75 if '_25' in label or '_75' in label else \
        0.5 if '_10' in label or '_90' in label else \
        0.25

    if 'positive' in label:
      color = POSITIVE_COLOR + (alpha,)
    else:
      color = NEGATIVE_COLOR + (alpha,)

    if '_50' in label:
      if 'positive' in label:
        label = 'positive'
      else:
        label = 'negative'
    else:
      label = None
    plt.plot(steps, values, color=color, label=label)

  plt.title(category)
  plt.legend(loc='upper left')
  plt.grid(True)
  plt.show()
