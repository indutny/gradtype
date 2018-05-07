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
RESTORE_FROM = os.environ.get('GRADTYPE_RESTORE')

LOG_DIR = os.path.join('.', 'logs', RUN_NAME)

# Maximum number of epochs to run for
MAX_EPOCHS = 500000

# Validate every epoch:
VALIDATE_EVERY = 5

# Number of sequences per category in batch
BATCH_SIZE = 16

# Learning rate
LR = 0.03

# Number of categories in each epoch
K = 64

#
# Load dataset
#

train_dataset, validate_dataset = dataset.load(train_overlap=4)

#
# Initialize model
#

input_shape = (None, dataset.MAX_SEQUENCE_LEN,)

codes = tf.placeholder(tf.int32, shape=input_shape, name='codes')
deltas = tf.placeholder(tf.float32, shape=input_shape, name='deltas')
category_count = tf.placeholder(tf.int32, shape=(), name='category_count')
training = tf.placeholder(tf.bool, shape=(), name='training')

model = Model(training=training)

output = model.build(codes, deltas)
t_metrics = model.get_metrics(output, category_count, BATCH_SIZE)

#
# Initialize optimizer
#

with tf.variable_scope('optimizer'):
  optimizer = tf.train.MomentumOptimizer(LR, momentum=0.9)
  t_reg_loss = tf.losses.get_regularization_loss()
  t_loss = t_metrics['loss'] + t_reg_loss
  variables = tf.trainable_variables()
  grads = tf.gradients(t_loss, variables)
  grads, t_grad_norm = tf.clip_by_global_norm(grads, 7.0)
  grads = list(zip(grads, variables))
  train = optimizer.apply_gradients(grads_and_vars=grads)

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

saver = tf.train.Saver(max_to_keep=10000, name=RUN_NAME)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  if RESTORE_FROM != None:
    print('Restoring from "{}"'.format(RESTORE_FROM))
    saver.restore(sess, RESTORE_FROM)

  step = 0
  for epoch in range(0, MAX_EPOCHS):
    train_batches = dataset.gen_hard_batches(train_dataset, \
        k=K,
        batch_size=BATCH_SIZE)

    saver.save(sess, LOG_DIR, global_step=step)
    print('Epoch {}'.format(epoch))
    for batch in train_batches:
      tensors = [ train, t_metrics, t_reg_loss, t_grad_norm ]
      _, metrics, reg_loss, grad_norm = sess.run(tensors, feed_dict={
        codes: batch['codes'],
        deltas: batch['deltas'],
        category_count: K,
        training: True,
      })
      metrics['regularization_loss'] = reg_loss
      metrics['grad_norm'] = grad_norm

      log_summary('train', metrics, step)
      step += 1
    writer.flush()

    if epoch % VALIDATE_EVERY != 0:
      continue

    validate_batches = dataset.gen_hard_batches(validate_dataset, \
        k=K,
        batch_size=BATCH_SIZE)

    print('Validation...')
    mean_metrics = None
    for batch in validate_batches:
      v_metrics = sess.run(t_metrics, feed_dict={
        codes: batch['codes'],
        deltas: batch['deltas'],
        category_count: K,
        training: False,
      })

      if mean_metrics is None:
        mean_metrics = {}
        for key in v_metrics:
          mean_metrics[key] = []

      for key in v_metrics:
        mean_metrics[key].append(v_metrics[key])

    for key in mean_metrics:
      mean_metrics[key] = np.mean(mean_metrics[key])

    log_summary('validate', mean_metrics, step)
    writer.flush()
