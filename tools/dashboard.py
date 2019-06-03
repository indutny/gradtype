import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import sys

# Internal
import binencoding

category = 'train' if len(sys.argv) < 3 else sys.argv[2]

fig = plt.figure(1, figsize=(10, 10))

raw = binencoding.load(sys.argv[1])

if 'step' in raw:
  step = 'step {}'.format(raw['step'])
else:
  step = 'unknown'

def hist(ax, category, positive, negative):
  # the histogram of the data
  ax.hist(positive, 500, color='green', density=True, alpha=0.5)
  ax.hist(negative, 500, color='red', density=True, alpha=0.5)

  ax.set_title(category)
  ax.set_xlabel('Distance')
  ax.set_ylabel('Percentage')
  ax.grid(True)

def roc(ax, category, positive, negative):
  y_true = [ 1 ] * len(positive) + [ 0 ] * len(negative)
  y_score = -np.concatenate([ positive, negative ])

  fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_score)
  auc = sklearn.metrics.auc(fpr, tpr)

  ax.plot(fpr, tpr, color='green' if category is 'validate' else 'blue',
      label='{}, AUC {:0.3f}'.format(category, auc))

def finish_roc(ax):
  ax.set_title('ROC')

  ax.set_xscale('log')
  ax.set_xlim([0.001, 1.0])
  ax.set_ylim([0.7, 1.0])
  ax.set_xlabel('False Positive Rate')
  ax.set_ylabel('True Positive Rate')
  ax.legend(loc='lower right')

  ax.grid(True, which='major', linestyle='-')
  ax.grid(True, which='minor', linestyle='--')

(train_hist, roc_ax), (validate_hist, aux) = fig.subplots(2, 2)

aux.set_title('{} / {}'.format(sys.argv[1], step))

for category in [ 'train', 'validate' ]:
  data = raw[category]

  positive = np.array(data['positives'])
  negative = np.array(data['negatives'])

  hist(train_hist if category == 'train' else validate_hist, \
      category, positive, negative)
  roc(roc_ax, category, positive, negative)

finish_roc(roc_ax)

fig.canvas.set_window_title('GradType')
fig.set_tight_layout(tight=True)
plt.show()
