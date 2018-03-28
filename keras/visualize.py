import matplotlib
matplotlib.use('Agg')

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn.decomposition

# Internal
import dataset

COLOR_MAP = plt.cm.tab20b

def pca(model, datasets, epoch):
  try:
    os.makedirs('./images/pca')
  except:
    None

  fig = plt.figure(1, figsize=(8, 6))
  ax = Axes3D(fig, elev=-150, azim=110)
  pca = sklearn.decomposition.PCA(n_components=3)

  ax.set_xlim(left=-1.2, right=1.2)
  ax.set_ylim(bottom=-1.2, top=1.2)
  ax.set_zlim(bottom=-1.2, top=1.2)

  res = dataset.apply_model(model, datasets)

  # Fit coordinates
  for ds_coords in res:
    pca.fit(ds_coords)

  # Transform coordinates and print labels
  colors = []
  all_x = []
  all_y = []
  all_z = []
  for i in range(0, len(res)):
    label = dataset.LABELS[i]
    ds_coords = pca.transform(res[i])

    x = ds_coords[:, 0]
    y = ds_coords[:, 1]
    z = ds_coords[:, 2]

    ax.text3D(x.mean(), y.mean(), z.mean(),
        label,
        fontsize=6,
        color=COLOR_MAP(i),
        horizontalalignment='center',
        bbox=dict(alpha=.1, edgecolor=COLOR_MAP(i), facecolor='w'))

    colors += [ i ] * len(x)
    all_x.append(x)
    all_y.append(y)
    all_z.append(z)

  all_x = np.concatenate(all_x)
  all_y = np.concatenate(all_y)
  all_z = np.concatenate(all_z)

  ax.scatter(all_x, all_y, all_z, c=colors,
             cmap=COLOR_MAP, edgecolor='k', s=5, alpha=0.8, linewidths=0.0,
             edgecolors='none')

  fname = './images/pca/{:08d}.png'.format(epoch)
  plt.savefig(fname=fname)
  print("Saved image to " + fname)
