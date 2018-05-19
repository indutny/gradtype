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

# Learning rate
LR = 0.01

BPTT_DEPTH = 64

#
# Load dataset
#

print('Loading data...')

loaded = dataset.load(mode='regression', train_overlap=None)
train_dataset = loaded['train']
validate_dataset = loaded['validate']

train_flat_dataset, train_weights = dataset.flatten_dataset(train_dataset)
validate_flat_dataset, validate_weights = \
    dataset.flatten_dataset(validate_dataset)

#
# Initialize model
#

input_shape = (None, None, dataset.MAX_CHAR + 1, )

rows = tf.placeholder(tf.float32, shape=input_shape, name='rows')
categories = tf.placeholder(tf.int32, shape=(None,), name='categories')
weights = tf.placeholder(tf.float32, shape=(None,), name='weights')
training = tf.placeholder(tf.bool, shape=(), name='training')

print('Building model...')

model = Model(training=False)

initial_states = model.initial_states(tf.shape(rows)[0])
states = model.create_states()

output, output_states = model.build(rows, states)

t_metrics, t_summary = model.get_regression_metrics(output, categories, weights)

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

grad_summary = []
for grad, var in grads:
  grad_summary.append(tf.summary.histogram(var.name + '/grad', grad))

t_summary = tf.summary.merge([ t_summary ] + grad_summary)

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

print('Starting...')

# TODO(indutny): move it to model
def run_batch(kind, batch, batch_weights, tensors):
  feed_dict = {
    rows: batch['rows'][:, :-BPTT_DEPTH],
    categories: batch['categories'],
    weights: batch_weights,
    training: kind is 'train',
  }
  model.assign_states(feed_dict, states, sess.run(initial_states,
      feed_dict=feed_dict))

  saved_states = sess.run(output_states, feed_dict=feed_dict)

  feed_dict[rows] = batch['rows'][:, -BPTT_DEPTH:]
  model.assign_states(feed_dict, states, saved_states)

  return sess.run(tensors, feed_dict=feed_dict)

with tf.Session() as sess:
  print('Initializing variables...')
  sess.run(tf.global_variables_initializer())

  if RESTORE_FROM != None:
    print('Restoring from "{}"'.format(RESTORE_FROM))
    saver.restore(sess, RESTORE_FROM)

  step = 0
  for epoch in range(0, MAX_EPOCHS):
    print('Generating batches...')
    train_batches = dataset.gen_regression(train_flat_dataset)
    validate_batches = dataset.gen_regression(validate_flat_dataset, \
        batch_size=len(validate_flat_dataset))

    print('Validation...')
    mean_metrics = None
    for batch in validate_batches:
      metrics, summary = run_batch('validate', batch, validate_weights,
          [ t_metrics, t_summary ])
      writer.add_summary(summary, step)

      if mean_metrics is None:
        mean_metrics = {}
        for key in metrics:
          mean_metrics[key] = []

      for key in metrics:
        mean_metrics[key].append(metrics[key])

    for key in mean_metrics:
      mean_metrics[key] = np.mean(mean_metrics[key])

    log_summary('validate', mean_metrics, step)

    saver.save(sess, LOG_DIR, global_step=step)
    print('Epoch {}'.format(epoch))
    for batch in train_batches:
      tensors = [ train, t_metrics, t_reg_loss, t_grad_norm ]
      _, metrics, reg_loss, grad_norm = run_batch('train', batch,
          validate_weights, tensors)
      metrics['regularization_loss'] = reg_loss
      metrics['grad_norm'] = grad_norm
      log_summary('train', metrics, step)

      step += 1

    writer.flush()
