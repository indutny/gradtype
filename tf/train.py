import os
import time

import numpy as np
import tensorflow as tf

# Internal
import dataset
from model import Model

RUN_NAME = os.environ.get('GRADTYPE_RUN')
if RUN_NAME is None:
  RUN_NAME = time.asctime()
LOG_DIR = os.path.join('.', 'logs', RUN_NAME)
SAVES_DIR = os.path.join('.', 'out', RUN_NAME)

# Maximum number of epochs to run for
MAX_EPOCHS = 500000

# Validate every epoch:
VALIDATE_EVERY = 5

# Number of sequences per category in batch
BATCH_SIZE = 32

#
# Load dataset
#

train_dataset, validate_dataset = dataset.load()

#
# Initialize model
#

model = Model(BATCH_SIZE)

input_shape = (None, dataset.MAX_SEQUENCE_LEN,)

codes = tf.placeholder(tf.int32, shape=input_shape, name='codes')
deltas = tf.placeholder(tf.float32, shape=input_shape, name='deltas')
category_count = tf.placeholder(tf.int32, (), name='category_count')

output = model.build(codes, deltas, category_count)
metrics = model.get_metrics(output, category_count)

#
# Initialize optimizer
#

optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(metrics['loss'] + \
    tf.losses.get_regularization_loss())

#
# TensorBoard
#

writer = tf.summary.FileWriter(LOG_DIR)
writer.add_graph(tf.get_default_graph())

def log_summary(prefix, metrics, step):
  summary = tf.Summary()
  for key in metrics:
    value = metrics[key]
    summary.value.add(tag='{}/{}'.format(prefix, key), simple_value=value)
  writer.add_summary(summary, step)
  writer.flush()

saver = tf.train.Saver(max_to_keep=10000, name=RUN_NAME)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  step = 0
  for epoch in range(0, MAX_EPOCHS):
    train_batches = dataset.gen_batches(train_dataset, batch_size=BATCH_SIZE)

    saver.save(sess, 'gradtype', global_step=step)
    print('Epoch {}'.format(epoch))
    for batch in train_batches:
      reg_loss = tf.losses.get_regularization_loss()
      tensors = [ train, metrics, reg_loss ]
      _, t_metrics, reg_loss = sess.run(tensors, feed_dict={
        codes: batch['codes'],
        deltas: batch['deltas'],
        category_count: len(train_dataset),
      })
      t_metrics['regularization_loss'] = reg_loss

      log_summary('train', t_metrics, step)
      step += 1

    if epoch % VALIDATE_EVERY != 0:
      continue

    validate_batches = dataset.gen_batches(validate_dataset, \
        batch_size=BATCH_SIZE)

    print('Validation...')
    mean_metrics = None
    for batch in validate_batches:
      v_metrics = sess.run(metrics, feed_dict={
        codes: batch['codes'],
        deltas: batch['deltas'],
        category_count: len(validate_dataset),
      })

      if mean_metrics is None:
        for key in v_metrics:
          mean_metrics[key] = []

      for key in v_metrics:
        mean_metrics[key].append(v_metrics[key])

    for key in mean_metrics:
      mean_metrics[key] = np.mean(mean_metrics[key])

    log_summary('validate', mean_metrics, step)
