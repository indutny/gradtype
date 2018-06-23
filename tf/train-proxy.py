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
SAVE_DIR = os.path.join('.', 'saves', RUN_NAME)

ADVERSARIAL_COUNT = None

# Number of sequences per batch
BATCH_SIZE = 2048

# Maximum number of epochs to run for
MAX_EPOCHS = 500000

# Validate every epoch:
VALIDATE_EVERY = 1

SAVE_EVERY = 10

# Learning rate
LR = 0.01

# Number of categories in each epoch
K = 64

#
# Load dataset
#

loaded = dataset.load(overlap=4)
train_dataset = loaded['train']
train_mask = loaded['train_mask']
validate_dataset = loaded['validate']
validate_mask = loaded['validate_mask']
category_count = loaded['category_count']

train_flat_dataset, train_weights = dataset.flatten_dataset(train_dataset)
validate_flat_dataset, validate_weights = \
    dataset.flatten_dataset(validate_dataset)
validate_batches = dataset.gen_regression(validate_flat_dataset, \
    batch_size=len(validate_flat_dataset))

#
# Initialize model
#

input_shape = (None, dataset.MAX_SEQUENCE_LEN,)

types = tf.placeholder(tf.float32, shape=input_shape, name='types')
codes = tf.placeholder(tf.int32, shape=input_shape, name='codes')
deltas = tf.placeholder(tf.float32, shape=input_shape, name='deltas')
sequence_lens = tf.placeholder(tf.int32, shape=(None,), name='sequence_lens')
training = tf.placeholder(tf.bool, shape=(), name='training')
categories = tf.placeholder(tf.int32, shape=(None,), name='categories')
category_mask = tf.placeholder(tf.bool, shape=(category_count,),
    name='category_mask')
weights = tf.placeholder(tf.float32, shape=(None,), name='weights')

model = Model(training=training)

output = model.build(types, codes, deltas, sequence_lens)
t_metrics = model.get_proxy_loss(output, categories, weights, category_count,
    category_mask, ADVERSARIAL_COUNT)
t_val_metrics = model.get_proxy_val_metrics(output, categories, weights,
    category_count, category_mask)

global_step_t = tf.Variable(0, trainable=False, name='global_step')
update_global_step_t = global_step_t.assign_add(1)

#
# Initialize optimizer
#

with tf.variable_scope('optimizer'):
  optimizer = tf.train.MomentumOptimizer(LR, momentum=0.9)
  t_reg_loss = tf.losses.get_regularization_loss()
  t_loss = t_metrics['loss'] + t_reg_loss
  variables = tf.trainable_variables()
  grads = tf.gradients(t_loss, variables)
  grads, t_grad_norm = tf.clip_by_global_norm(grads, 2.0)
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

  step = tf.train.global_step(sess, global_step_t)

  for epoch in range(0, MAX_EPOCHS):
    train_batches = dataset.gen_regression(train_flat_dataset,
        adversarial_count=ADVERSARIAL_COUNT, batch_size=BATCH_SIZE)

    if epoch % SAVE_EVERY == 0:
      print('Saving...')
      saver.save(sess, os.path.join(SAVE_DIR, '{:08d}'.format(step)))

    print('Epoch {}'.format(epoch))
    for batch in train_batches:
      tensors = [ train, update_global_step_t, t_metrics, t_reg_loss,
          t_grad_norm ]
      _, _, metrics, reg_loss, grad_norm = sess.run(tensors, feed_dict={
        types: batch['types'],
        codes: batch['codes'],
        deltas: batch['deltas'],
        sequence_lens: batch['sequence_lens'],
        categories: batch['categories'],
        category_mask: train_mask,
        weights: train_weights,
        training: True,
      })
      metrics['regularization_loss'] = reg_loss
      metrics['grad_norm'] = grad_norm

      step += 1
      log_summary('train', metrics, step)

    print('Validation...')
    mean_metrics = None
    for batch in validate_batches:
      metrics = sess.run(t_val_metrics, feed_dict={
        types: batch['types'],
        codes: batch['codes'],
        deltas: batch['deltas'],
        sequence_lens: batch['sequence_lens'],
        categories: batch['categories'],
        category_mask: validate_mask,
        weights: validate_weights,
        training: False,
      })

      if mean_metrics is None:
        mean_metrics = {}
        for key in metrics:
          mean_metrics[key] = []

      for key in metrics:
        mean_metrics[key].append(metrics[key])

    for key in mean_metrics:
      mean_metrics[key] = np.mean(mean_metrics[key])

    log_summary('validate', mean_metrics, step)
    writer.flush()
