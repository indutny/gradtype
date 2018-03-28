import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn.decomposition

def pca(model, datasets, epoch):
  try:
    os.mkdir('./images')
  except:
    None
  try:
    os.mkdir('./images/pca')
  except:
    None
  colors = []
  coordinates = []
  for i in range(0, len(datasets)):
    ds = datasets[i]
    codes = []
    deltas = []
    for seq in ds:
      codes.append(seq['codes'])
      deltas.append(seq['deltas'])
    codes = np.array(codes)
    deltas = np.array(deltas)
    result = model.predict(x={ 'codes': codes, 'deltas': deltas })
    for c in result:
      colors.append(i)
      coordinates.append(c)

  # Reduce features vector dimension
  reduced = sklearn.decomposition.PCA(n_components=3).fit_transform(coordinates)

  fig = plt.figure(1, figsize=(8, 6))
  ax = Axes3D(fig, elev=-150, azim=110)
  ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=colors,
             cmap=plt.cm.Set1, edgecolor='k', s=5, alpha=0.5, linewidths=0.0,
             edgecolors='none')
  plt.savefig(fname='./images/pca/' + str(epoch) + '.png')
