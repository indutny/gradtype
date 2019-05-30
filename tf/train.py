import os
import time

import numpy as np
import tensorflow as tf

# Internal
import dataset
from model import Model

RUN_NAME = os.environ.get('GRADTYPE_RUN')
if RUN_NAME is None:
  RUN_NAME = 'gradtype'
RESTORE_FROM = os.environ.get('GRADTYPE_RESTORE')

LOG_DIR = os.path.join('.', 'logs', RUN_NAME)
SAVE_DIR = os.path.join('.', 'saves', RUN_NAME)

# Number of sequences per batch
BATCH_SIZE = 256

# Maximum number of epochs to run for
MAX_EPOCHS = 500000

# Validate every epoch:
VALIDATE_EVERY = 10

SAVE_EVERY = 100

# Learning rate
LR = 0.001

AUTO = os.environ.get('GRADTYPE_AUTO') == 'on'

#
# Load dataset
#

loaded = dataset.load(overlap=4)
train_dataset = loaded['train']
train_mask = loaded['train_mask']
validate_dataset = loaded['validate']
validate_mask = loaded['validate_mask']
category_count = loaded['category_count']

train_batches_gen = dataset.gen_regression(train_dataset,
    batch_size=BATCH_SIZE)
validate_batches = next(
    dataset.gen_regression(validate_dataset, batch_size=None))

#
# Initialize model
#

input_shape = (None, dataset.MAX_SEQUENCE_LEN,)

holds = tf.placeholder(tf.float32, shape=input_shape, name='holds')
codes = tf.placeholder(tf.int32, shape=input_shape, name='codes')
deltas = tf.placeholder(tf.float32, shape=input_shape, name='deltas')
sequence_lens = tf.placeholder(tf.int32, shape=(None,), name='sequence_lens')
training = tf.placeholder(tf.bool, shape=(), name='training')
categories = tf.placeholder(tf.int32, shape=(None,), name='categories')
category_mask = tf.placeholder(tf.bool, shape=(category_count,),
    name='category_mask')

model = Model(training=training)

global_step_t = tf.Variable(0, trainable=False, name='global_step')
update_global_step_t = global_step_t.assign_add(1)

output = model.build(holds, codes, deltas, sequence_lens, auto=AUTO)
if not auto:
  aux_output = model.build(holds, codes, deltas, sequence_lens, auto=AUTO)
  aux_metrics = model.get_proxy_loss(aux_output, categories, category_count,
      category_mask, tf.cast(global_step_t, dtype=tf.float32))
else:
  aux_output = model.build(holds, codes, deltas, sequence_lens, auto=True)

if AUTO:
  t_metrics = model.get_auto_loss(holds, deltas, output)
  t_val_metrics = t_metrics
else:
  t_metrics = model.get_proxy_loss(output, categories, category_count,
      category_mask, tf.cast(global_step_t, dtype=tf.float32))
  t_val_metrics = model.get_proxy_val_metrics(output, categories,
      category_count, category_mask)

#
# Initialize optimizer
#

with tf.variable_scope('optimizer'):
  t_lr = tf.constant(LR, dtype=tf.float32)
  if AUTO:
    power = tf.floor(tf.cast(global_step_t, dtype=tf.float32) / 50000.0)
    power = tf.minimum(3.0, power)
    t_lr /= 10.0 ** power
    t_metrics['lr'] = t_lr
  optimizer = tf.train.AdamOptimizer(t_lr)
  t_reg_loss = tf.losses.get_regularization_loss()
  t_loss = t_metrics['loss'] + t_reg_loss
  variables = tf.trainable_variables()
  unclipped_grads = tf.gradients(t_loss, variables)
  grads, t_grad_norm = tf.clip_by_global_norm(unclipped_grads, 1000.0)
  for (grad, var) in zip(unclipped_grads, variables):
    t_metrics['grad_' + var.name] = tf.norm(grad) / (t_grad_norm + 1e-23)

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
    start_time = time.time()
    train_batches = next(train_batches_gen)
    end_time = time.time()
    print('Generated batches in: {}'.format(end_time - start_time))

    if epoch % SAVE_EVERY == 0:
      print('Saving...')
      saver.save(sess, os.path.join(SAVE_DIR, '{:08d}'.format(step)))

    print('Epoch {}, step {}'.format(epoch, step))
    start_time = time.time()
    for batch in train_batches:
      tensors = [ train, update_global_step_t, t_metrics, t_reg_loss,
          t_grad_norm ]
      train_feed = {
          holds: batch['holds'],
          codes: batch['codes'],
          deltas: batch['deltas'],
          sequence_lens: batch['sequence_lens'],
          categories: batch['categories'],
          category_mask: train_mask,
          training: True,
        }
      try:
        _, _, metrics, reg_loss, grad_norm = sess.run(tensors,
            feed_dict=train_feed)
      except tf.errors.InvalidArgumentError:
        # Catch NaN and inf global norm
        print('got invalid argument error, printing gradients')
        for (grad, var) in zip(unclipped_grads, variables):
          print('{}: {}'.format(var.name, sess.run(tf.reduce_mean(grad), feed_dict=train_feed)))
        raise
      metrics['regularization_loss'] = reg_loss
      metrics['grad_norm'] = grad_norm

      step += 1
      log_summary('train', metrics, step)
    end_time = time.time()
    print('Mean batch time: {}'.format(
      (end_time - start_time) / len(train_batches)))

    if epoch % VALIDATE_EVERY == 0:
      print('Validation...')
      mean_metrics = None
      for batch in validate_batches:
        metrics = sess.run(t_val_metrics, feed_dict={
          holds: batch['holds'],
          codes: batch['codes'],
          deltas: batch['deltas'],
          sequence_lens: batch['sequence_lens'],
          categories: batch['categories'],
          category_mask: validate_mask,
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
