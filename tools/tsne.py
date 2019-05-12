import matplotlib
import sys
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.axes as axes
import numpy as np
import sklearn.decomposition
from sklearn.manifold import TSNE

COLOR_MAP = plt.cm.gist_rainbow
SEED = 0x37255c25

CATEGORIES = {}

def to_color(index):
  return index / len(CATEGORIES)

def visualize(entries):
  fig = plt.figure(1, figsize=(8, 6))

  axes = plt.gca()

  if True:
    decomp = TSNE(n_components=2, verbose=2, random_state=SEED,
        perplexity=30, n_iter=100000)
  else:
    decomp = sklearn.decomposition.PCA(n_components=2, random_state=SEED)
  coords = decomp.fit_transform([ e['features'] for e in entries ])

  for e, coords in zip(entries, coords):
    e['coords'] = coords
    category = e['category']
    if category in CATEGORIES:
      index = CATEGORIES[category]
    else:
      index = len(CATEGORIES)
      CATEGORIES[category] = index
    e['index'] = index

  legend = []
  for label, index in CATEGORIES.items():
    color = COLOR_MAP(to_color(index))
    legend.append(mpatches.Patch(color=color, label=label))
  # axes.legend(handles=legend, fontsize=8)

  for e in entries:
    label = e['category']
    index = e['index']

    x = e['coords'][0]
    y = e['coords'][1]

    color = COLOR_MAP(to_color(index))

    marker = 'o'
    size = 16
    alpha = 0.6
    plt.scatter(x, y, c=[ color ], marker=marker,
                edgecolor='k', s=size, alpha=alpha, linewidths=0.0,
                edgecolors='none')

  plt.show()

with open(sys.argv[1]) as input:
  data = json.load(input)
  visualize(data['train'] + data['validate'])
