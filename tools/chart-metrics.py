import csv
import numpy as np
import matplotlib.pyplot as plt
import sys

category = 'train' if len(sys.argv) < 3 else sys.argv[2]

ALPHA = 0.2
NEGATIVE_COLOR = (1.0, 0.098039215686275, 0.223529411764706,)
POSITIVE_COLOR = (0.219607843137255, 0.698039215686275, 0.317647058823529,)

fig = plt.figure(1, figsize=(8, 8))

for category in [ 'train', 'validate' ]:
  with open(sys.argv[1]) as input:
    reader = csv.reader(input, delimiter=',')
    labels = next(reader)

    steps = []
    data = {}
    for row in reader:
      steps.append(int(row[1]))
      for value, label in zip(row[2:], labels[2:]):
        if not category in label:
          continue

        if not label in data:
          data[label] = []
        data[label].append(float(value))

  rev_steps = steps.copy()
  rev_steps.reverse()
  fill_steps = steps + rev_steps

  plt.subplot(2, 1, 1 if category is 'train' else 2)

  def fill(forward, backward, color, alpha=1.0, label=None):
    color += (alpha,)
    backward_rev = backward.copy()
    backward_rev.reverse()
    plt.fill(fill_steps, forward + backward_rev,
        color=color, label=label)

  fill(data[category + '/positive_5'], data[category + '/positive_95'],
      POSITIVE_COLOR, ALPHA)
  fill(data[category + '/positive_10'], data[category + '/positive_90'],
      POSITIVE_COLOR, ALPHA)
  fill(data[category + '/positive_25'], data[category + '/positive_75'],
      POSITIVE_COLOR, ALPHA)
  plt.plot(steps, data[category + '/positive_50'], color=POSITIVE_COLOR,
      label='positive')

  fill(data[category + '/negative_5'], data[category + '/negative_95'],
      NEGATIVE_COLOR, ALPHA)
  fill(data[category + '/negative_10'], data[category + '/negative_90'],
      NEGATIVE_COLOR, ALPHA)
  fill(data[category + '/negative_25'], data[category + '/negative_75'],
      NEGATIVE_COLOR, ALPHA)

  plt.plot(steps, data[category + '/negative_50'], color=NEGATIVE_COLOR,
      label='negative')

  plt.title(category)
  plt.legend(loc='upper left')
  plt.grid(True)

plt.tight_layout()
plt.show()
