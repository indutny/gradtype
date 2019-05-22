import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import sys

category = 'train' if len(sys.argv) < 3 else sys.argv[2]

with open(sys.argv[1]) as f:
  data = json.load(f)
  data = data[category]

  positive = data['positives']
  negative = data['negatives']

  y_true = [ 1 ] * len(positive) + [ 0 ] * len(negative)
  y_score = -np.array(positive + negative)

  fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_score)
  auc = sklearn.metrics.auc(fpr, tpr)

  fig = plt.figure(1, figsize=(8, 8))

  plt.plot(fpr, tpr, color='green', label='AUC %0.3f' % auc)
  plt.xscale('log')

  plt.title(category)
  plt.xlim([0.001, 1.0])
  plt.ylim([0.7, 1.0])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc='lower right')

  plt.grid(True, which='major', linestyle='-')
  plt.grid(True, which='minor', linestyle='--')
  plt.show()
