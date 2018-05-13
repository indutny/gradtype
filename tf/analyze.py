import sys
import numpy as np
import tensorflow as tf

# Internal
import dataset
from model import Model

model = Model(training=False)

input_shape = (None, dataset.MAX_SEQUENCE_LEN,)

p_codes = tf.placeholder(tf.int32, shape=input_shape, name='codes')
p_deltas = tf.placeholder(tf.float32, shape=input_shape, name='deltas')

output = model.build(p_codes, p_deltas)

def group_coords(dataset):
  result = {}
  for seq in dataset:
    category = seq['category']
    if category in result:
      result[category]['coords'].append(seq['features'])
    else:
      result[category] = { 'coords': [ seq['features'] ] }

  for group in result.values():
    group['coords'] = np.array(group['coords'])

  return result

def compute_centers(groups):
  for group in groups.values():
    group['center'] = np.mean(group['coords'], axis=0)

def compute_positives(groups):
  for group in groups.values():
    positives = group['coords'] - np.expand_dims(group['center'], 0)
    positives **= 2
    positives = np.sqrt(np.sum(positives, axis=-1))
    group['positive_5'] = np.percentile(positives, 5)
    group['positive_25'] = np.percentile(positives, 25)
    group['positive_50'] = np.percentile(positives, 50)
    group['positive_75'] = np.percentile(positives, 75)
    group['positive_95'] = np.percentile(positives, 95)

def compute_negatives(groups):
  for i, group in groups.items():
    neg_centers = []
    for j in groups:
      if i == j:
        continue
      neg_centers.append(groups[j]['center'])

    negatives = np.expand_dims(group['coords'], 1) - \
        np.expand_dims(neg_centers, 0)
    negatives **= 2
    negatives = np.sqrt(np.sum(negatives, axis=-1))

    group['negative_95'] = np.percentile(negatives, 95)
    group['negative_75'] = np.percentile(negatives, 75)
    group['negative_50'] = np.percentile(negatives, 50)
    group['negative_25'] = np.percentile(negatives, 25)
    group['negative_5'] = np.percentile(negatives, 5)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  saver = tf.train.Saver(max_to_keep=0, name='analyze')
  restore = sys.argv[1]
  if restore.endswith('.index'):
    restore = restore[:-6]
  saver.restore(sess, restore)

  train_dataset, validate_dataset = dataset.load()

  train_dataset = dataset.flatten_dataset(train_dataset)
  validate_dataset = dataset.flatten_dataset(validate_dataset)

  codes = []
  deltas = []

  for seq in train_dataset:
    codes.append(seq['codes'])
    deltas.append(seq['deltas'])

  for seq in validate_dataset:
    codes.append(seq['codes'])
    deltas.append(seq['deltas'])

  features = sess.run(output, feed_dict={
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

  val_groups = group_coords(validate_features)
  compute_centers(val_groups)
  compute_positives(val_groups)
  compute_negatives(val_groups)

  for cat, group in val_groups.items():
    print('Category: {}'.format(cat))

    print('  positive_5: {}'.format(group['positive_5']))
    print('  positive_25: {}'.format(group['positive_25']))
    print('  positive_50: {}'.format(group['positive_50']))
    print('  positive_75: {}'.format(group['positive_75']))
    print('  positive_95: {}'.format(group['positive_95']))

    print('  negative_5: {}'.format(group['negative_5']))
    print('  negative_25: {}'.format(group['negative_25']))
    print('  negative_50: {}'.format(group['negative_50']))
    print('  negative_75: {}'.format(group['negative_75']))
    print('  negative_95: {}'.format(group['negative_95']))
