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
import sklearn.manifold

# Internal
import dataset
import model as gradtype_model

COLOR_MAP = plt.cm.gist_rainbow

def to_color(index):
  return index / (len(dataset.LABELS) - 1)

def pca(train_coords, validate_coords, fname):
  fig = plt.figure(1, figsize=(8, 6))
  pca = sklearn.decomposition.PCA(n_components=50, random_state=0x7ed1ae6e)
  tsne = sklearn.manifold.TSNE(n_components=2, random_state=0x18e72aad)

  # ax.set_xlim(left=-0.9, right=0.9)
  # ax.set_ylim(bottom=-0.9, top=0.9)
  # ax.set_zlim(bottom=-0.9, top=0.9)

  # Fit coordinates
  all_coords = pca.fit_transform(np.concatenate(train_coords + validate_coords))
  exit(0)

  for coords in [ train_coords, validate_coords ]:
    np.random.permutation()

  # Print labels
  # if len(dataset.LABELS) < 32:
  #   legend = []
  #   for i in range(0, len(dataset.LABELS)):
  #     color = COLOR_MAP(to_color(i))
  #     legend.append(mpatches.Patch(color=color, label=dataset.LABELS[i]))
  #   plt.legend(handles=legend, fontsize=8)

  pairs = {}

  all_coords = [ train_coords, validate_coords ]
  for coord_type in range(0, len(all_coords)):
    colors = []
    all_x = []
    all_y = []

    coords = all_coords[coord_type]
    is_train = coord_type is 0
    for i in range(0, len(coords)):
      ds_coords = pca.transform(coords[i])

      x = ds_coords[:, 0]
      y = ds_coords[:, 1]

      label = dataset.LABELS[i]
      color = COLOR_MAP(to_color(i))

      mean_x = x.mean()
      mean_y = y.mean()

      if pairs.get(label) is None:
        pairs[label] = { 'x': [], 'y': [] }

      pairs[label]['x'].append(mean_x)
      pairs[label]['y'].append(mean_y)

      marker = 'o' if is_train else '^'
      size = 24 if is_train else 32
      plt.scatter(mean_x, mean_y, c=color, marker=marker,
                  edgecolor='white', s=size, alpha=1.0)

      colors += [ color ] * len(x)
      all_x.append(x)
      all_y.append(y)

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    marker = 'o' if is_train else '^'
    size = 5 if is_train else 8
    plt.scatter(all_x, all_y, c=colors, marker=marker,
               edgecolor='k', s=size, alpha=0.1, linewidths=0.0,
               edgecolors='none')

  for i in range(0, len(dataset.LABELS)):
    entry = pairs.get(dataset.LABELS[i])
    if entry is None:
      continue
    if len(entry['x']) != 2:
      continue

    color = COLOR_MAP(to_color(i))
    plt.plot(entry['x'], entry['y'], color=color)

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

  train_datasets, validate_datasets = dataset.split(datasets)
  train_coords = dataset.evaluate_model(siamese, train_datasets)
  validate_coords = dataset.evaluate_model(siamese, validate_datasets)
  fname = sys.argv[2] if len(sys.argv) >= 3 else  None
  pca(train_coords, validate_coords, fname)
