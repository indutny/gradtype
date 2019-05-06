import sys
import logging

import numpy as np
import tensorflow as tf

# Internal
import dataset
from model import Model

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

SEED = 0x37255c25
PERCENTILES = [ 5, 10, 25, 50, 75, 90, 95 ]

def hist_dist(dataset):
  positives = []
  negatives = []
  total = len(dataset) * (len(dataset) - 1) / 2
  done = 0
  last_report = 0
  for i, seq in enumerate(dataset):
    seq_category = seq['category']
    seq_features = seq['features']
    for other in dataset[i + 1:]:
      other_category = other['category']
      other_features = other['features']

      distance = (seq_features - other_features) ** 2
      distance = np.sum(distance)
      distance = np.sqrt(distance)

      if seq_category == other_category:
        positives.append(distance)
      else:
        negatives.append(distance)

      done += 1
      if (done / total) >= last_report + 0.05:
        last_report = done / total
        logging.debug('Cross distance progress: {}'.format(last_report))

  positives = np.array(positives)
  negatives = np.array(negatives)

  results = []
  for values in [ negatives, positives ]:
    for q in PERCENTILES:
      results.append(np.nanpercentile(values, q))
  return results

model = Model(training=False)

input_shape = (None, dataset.MAX_SEQUENCE_LEN,)

p_codes = tf.placeholder(tf.int32, shape=input_shape, name='codes')
p_holds = tf.placeholder(tf.float32, shape=input_shape, name='holds')
p_deltas = tf.placeholder(tf.float32, shape=input_shape, name='deltas')

output = model.build(p_holds, p_codes, p_deltas)

global_step_t = tf.Variable(0, trainable=False, name='global_step')

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  logging.debug('Loading model...')

  saver = tf.train.Saver(max_to_keep=0, name='hist-dist')
  restore = sys.argv[1]
  if restore.endswith('.index'):
    restore = restore[:-6]
  saver.restore(sess, restore)

  logging.debug('Loading dataset...')

  loaded = dataset.load(overlap=8)
  train_dataset = loaded['train']
  validate_dataset = loaded['validate']

  logging.debug('Trimming dataset...')

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

  logging.debug('Gathering features...')

  step, features = sess.run([ global_step_t, output ], feed_dict={
    p_holds: holds,
    p_codes: codes,
    p_deltas: deltas,
  })

  logging.debug('Global step: {}'.format(step))

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

  logging.debug('Computing percentiles...')

  out_name = sys.argv[2] if len(sys.argv) >= 3 else None

  train_p = hist_dist(train_features)
  val_p = hist_dist(validate_features)

  result = [ 0, step ] + train_p + val_p
  print(','.join(str(p) for p in result))
