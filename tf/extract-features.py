import sys
import logging
import json

import numpy as np
import tensorflow as tf

# Internal
import dataset
from model import Model

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

SEED = 0x37255c25

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

  train_out = []
  for seq in train_dataset:
    train_out.append({
      'category': seq['category'],
      'features': [ float(x) for x in features[0] ],
    })
    features = features[1:]

  validate_out = []
  for seq in validate_dataset:
    validate_out.append({
      'category': seq['category'],
      'features': [ float(x) for x in features[0] ],
    })
    features = features[1:]

  out = { 'train': train_out, 'validate': validate_out }
  print(json.dumps(out))
