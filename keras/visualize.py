import matplotlib
import sys

# Do not display GUI only when generating output
if __name__ != '__main__' or len(sys.argv) >= 3:
  matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn.decomposition

# Internal
import dataset
import model as gradtype_model

COLOR_MAP = plt.cm.tab20b

def pca(coords, fname):
  fig = plt.figure(1, figsize=(8, 6))
  ax = Axes3D(fig, elev=-150, azim=110)
  pca = sklearn.decomposition.PCA(n_components=3, random_state=0x7ed1ae6e)

  ax.set_xlim(left=-1.2, right=1.2)
  ax.set_ylim(bottom=-1.2, top=1.2)
  ax.set_zlim(bottom=-1.2, top=1.2)

  # Fit coordinates
  pca.fit(np.concatenate(coords))

  # Transform coordinates and print labels
  colors = []
  all_x = []
  all_y = []
  all_z = []
  for i in range(0, len(coords)):
    label = dataset.LABELS[i]
    ds_coords = pca.transform(coords[i])

    x = ds_coords[:, 0]
    y = ds_coords[:, 1]
    z = ds_coords[:, 2]

    ax.text3D(x.mean(), y.mean(), z.mean(),
        label,
        fontsize=7,
        color=COLOR_MAP(i),
        horizontalalignment='center',
        bbox=dict(alpha=.3, edgecolor=COLOR_MAP(i), facecolor='w'))

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

  if fname == None:
    plt.show()
  else:
    plt.savefig(fname=fname)
    print("Saved image to " + fname)

if __name__ == '__main__':
  import sys

  datasets, sequence_len = dataset.parse()
  siamese, model = gradtype_model.create(sequence_len)
  model.load_weights(sys.argv[1])

  coordinates = dataset.evaluate_model(siamese, datasets)
  pca(coordinates, sys.argv[2] if len(sys.argv) >= 3 else  None)
