import csv
import numpy as np
import matplotlib.pyplot as plt
import sys

category = 'train' if len(sys.argv) < 3 else sys.argv[2]

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

    if 'negative' in label:
      color = '#0061ff' if '_50' in label else \
          '#68a2ff' if '_25' in label or '_75' in label else \
          '#89b6ff' if '_10' in label or '_90' in label else \
          '#adccff'
    else:
      color = '#fc0036' if '_50' in label else \
          '#ff7290' if '_25' in label or '_75' in label else \
          '#ff9eb3' if '_10' in label or '_90' in label else \
          '#ffc6d3'

    plt.plot(steps, values, color=color, label=label)

  plt.title(category)
  plt.legend(loc='upper left')
  plt.grid(True)
  plt.show()
