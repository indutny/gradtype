import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import sys

with open(sys.argv[1]) as f:
  fig = plt.figure(1, figsize=(8, 8))
  data = json.load(f)

  if 'step' in data:
    step = 'step {}'.format(data['step'])
  else:
    step = 'unknown'

  for category in [ 'train', 'validate' ]:
    sub_data = data[category]

    positive = sub_data['positives']
    negative = sub_data['negatives']

    y_true = [ 1 ] * len(positive) + [ 0 ] * len(negative)
    y_score = -np.array(positive + negative)

    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_score)
    auc = sklearn.metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, color='green' if category is 'validate' else 'blue',
        label='{}, AUC {:0.3f}'.format(category, auc))

  plt.title(sys.argv[1] + ' / ' + step)

  plt.xscale('log')
  plt.xlim([0.001, 1.0])
  plt.ylim([0.7, 1.0])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc='lower right')

  plt.grid(True, which='major', linestyle='-')
  plt.grid(True, which='minor', linestyle='--')
  plt.tight_layout()
  plt.show()
