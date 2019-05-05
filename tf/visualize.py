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
from sklearn.manifold import TSNE
import tensorflow as tf

# Internal
import dataset
from model import Model

COLOR_MAP = plt.cm.gist_rainbow
LABELS = dataset.load_labels()
SEED = 0x37255c25

def to_color(index):
  return index / (len(LABELS) - 1)

USE_TSNE = True

def pca(train, validate, fname=None):
  fig = plt.figure(1, figsize=(8, 6))
  if not USE_TSNE:
    pca = sklearn.decomposition.PCA(n_components=2, random_state=SEED)

  axes = plt.gca()
  if not USE_TSNE:
    axes.set_xlim([ -1.5, 1.5 ])
    axes.set_ylim([ -1.5, 1.5 ])

  if USE_TSNE:
    pca = sklearn.decomposition.PCA(n_components=50, random_state=SEED)
    tsne = TSNE(n_components=2, verbose=2, random_state=SEED)

  # Fit coordinates
  coords = pca.fit_transform([ seq['features'] for seq in (train + validate) ])
  if USE_TSNE:
    coords = tsne.fit_transform(coords)

  train_coords = coords[:len(train)]
  validate_coords = coords[len(train_coords):]

  for seq, coords in zip(train, train_coords):
    seq['coords'] = coords

  for seq, coords in zip(validate, validate_coords):
    seq['coords'] = coords

  for kind, dataset in zip([ 'train', 'validate' ], [ train, validate ]):
    colors = []
    all_x = []
    all_y = []

    for seq in dataset:
      category = seq['category']

      x = seq['coords'][0]
      y = seq['coords'][1]

      label = seq['label']
      color = COLOR_MAP(to_color(category))

      marker = 'o' if kind is 'train' else '^'
      size = 6 if kind is 'train' else 10
      alpha = 0.45 if kind is 'train' else 0.8
      plt.scatter(x, y, c=[ color ], marker=marker,
                  edgecolor='k', s=size, alpha=alpha, linewidths=0.0,
                  edgecolors='none')

  if fname == None:
    plt.show()
  else:
    plt.savefig(fname=fname, dpi='figure')
    print("Saved image to " + fname)

model = Model(training=False)

input_shape = (None, dataset.MAX_SEQUENCE_LEN,)

p_codes = tf.placeholder(tf.int32, shape=input_shape, name='codes')
p_holds = tf.placeholder(tf.float32, shape=input_shape, name='holds')
p_deltas = tf.placeholder(tf.float32, shape=input_shape, name='deltas')

output = model.build(p_holds, p_codes, p_deltas)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  saver = tf.train.Saver(max_to_keep=0, name='visualize')
  restore = sys.argv[1]
  if restore.endswith('.index'):
    restore = restore[:-6]
  saver.restore(sess, restore)

  loaded = dataset.load(overlap=8)
  train_dataset = loaded['train']
  validate_dataset = loaded['validate']

  train_dataset, _ = dataset.trim_dataset(train_dataset,
      random_state=SEED)
  validate_dataset, _ = dataset.trim_dataset(validate_dataset,
      random_state=SEED)

  train_dataset, _ = dataset.flatten_dataset(train_dataset,
      random_state=SEED)
  validate_dataset, _ = dataset.flatten_dataset(validate_dataset,
      random_state=SEED)

  holds = []
  codes = []
  deltas = []

  for seq in train_dataset:
    holds.append(seq['holds'])
    codes.append(seq['codes'])
    deltas.append(seq['deltas'])

  for seq in validate_dataset:
    holds.append(seq['holds'])
    codes.append(seq['codes'])
    deltas.append(seq['deltas'])

  features = sess.run(output, feed_dict={
    p_holds: holds,
    p_codes: codes,
    p_deltas: deltas,
  })

  train_features = []
  for seq in train_dataset:
    seq = dict(seq)
    seq.update({ 'features': features[0] })
    train_features.append(seq)
    features = features[1:]

  validate_features = []
  for seq in validate_dataset:
    seq = dict(seq)
    seq.update({ 'features': features[0] })
    validate_features.append(seq)
    features = features[1:]

  out_name = sys.argv[2] if len(sys.argv) >= 3 else None
  pca(train_features, validate_features, fname=out_name)
