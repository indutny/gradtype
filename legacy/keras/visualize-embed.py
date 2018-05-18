import matplotlib
import sys

# Do not display GUI only when generating output
if __name__ != '__main__' or len(sys.argv) >= 3:
  matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.axes as axes
import numpy as np
import sklearn.decomposition

# Internal
import dataset
import model as gradtype_model

COLOR_MAP = plt.cm.gist_rainbow
LABELS = [
  'pad',
  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
  'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
  ' ', ','
]

def to_color(index):
  return index / (dataset.MAX_CHAR + 2)

def pca(embedding, fname):
  fig = plt.figure(1, figsize=(8, 6))
  pca = sklearn.decomposition.PCA(n_components=2, random_state=0x7ed1ae6e)

  # Fit coordinates
  embedding = pca.fit_transform(embedding[0])

  # Print labels
  legend = []
  for i in range(0, len(LABELS)):
    color = COLOR_MAP(to_color(i))
    legend.append(mpatches.Patch(color=color, label=LABELS[i]))
  plt.legend(handles=legend, fontsize=8)

  colors = [ COLOR_MAP(to_color(i)) for i in range(0, len(embedding)) ]
  plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, marker='o',
             edgecolor='k', s=16, alpha=1, linewidths=0.0,
             edgecolors='none')

  if fname == None:
    plt.show()
  else:
    plt.savefig(fname=fname)
    print("Saved image to " + fname)

if __name__ == '__main__':
  import sys

  datasets = dataset.parse()
  siamese, _, _ = gradtype_model.create()
  siamese.load_weights(sys.argv[1])
  embedding = siamese.get_layer('embed').get_weights()

  fname = sys.argv[2] if len(sys.argv) >= 3 else None
  pca(embedding, fname)
