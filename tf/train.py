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

AUTO_CLIP = False

# See https://arxiv.org/pdf/1511.06807.pdf
STATIC_CLIP = 10.0
GRAD_NOISE_GAMMA = 0.55
GRAD_NOISE_ETA = 0.3

# Maximum number of epochs to run for
MAX_EPOCHS = 500000

# Validate every epoch:
VALIDATE_EVERY = 10

SAVE_EVERY = 100

# Learning rate
LR = 0.01

#
# Load dataset
#

loaded = dataset.load()
train_dataset = loaded['train']
train_mask = loaded['train_mask']
validate_dataset = loaded['validate']
validate_mask = loaded['validate_mask']
category_count = loaded['category_count']

train_batches_gen = dataset.gen_regression(train_dataset,
    batch_size=None, randomize=False)
validate_batches = next(
    dataset.gen_regression(validate_dataset, batch_size=None))

#
# Initialize model
#

input_shape = (None, dataset.MAX_SEQUENCE_LEN,)

grad_clip = tf.placeholder(tf.float32, shape=(), name='grad_clip')
grad_clip_lambda = 0.01
grad_clip_mul = 1.25
grad_clip_value = 1.0

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

output = model.build(holds, codes, deltas, sequence_lens)

t_metrics = model.get_proxy_loss(output, categories, category_count,
    category_mask, tf.cast(global_step_t, dtype=tf.float32))
t_val_metrics = model.get_proxy_val_metrics(output, categories,
    category_count, category_mask)

#
# Initialize optimizer
#

with tf.variable_scope('optimizer'):
  t_lr = tf.constant(LR, dtype=tf.float32)

  t_reg_loss = tf.losses.get_regularization_loss()
  t_loss = t_metrics['loss'] + t_reg_loss

  noise_dev = GRAD_NOISE_ETA
  noise_dev /= (1 + tf.cast(global_step_t, dtype=tf.float32)) \
      ** GRAD_NOISE_GAMMA
  noise_dev = tf.sqrt(noise_dev, name='noise_dev')

  t_metrics['regularization_loss'] = t_reg_loss
  t_metrics['noise_dev'] = noise_dev

  variables = tf.trainable_variables()
  def get_train(t_loss, t_metrics):
    unclipped_grads = tf.gradients(t_loss, variables)
    grads, t_grad_norm = tf.clip_by_global_norm(unclipped_grads, grad_clip)
    for (grad, var) in zip(unclipped_grads, variables):
      if not grad is None:
        t_metrics['grad_' + var.name] = tf.norm(grad) / (t_grad_norm + 1e-23)
    grads = [
        g + tf.random.normal(tf.shape(g), mean=0.0, stddev=noise_dev)
        for g in grads
    ]
    grads = list(zip(grads, variables))

    t_metrics['grad_norm'] = t_grad_norm
    t_metrics['grad_clip'] = grad_clip

    optimizer = tf.train.AdamOptimizer(t_lr)
    return optimizer.apply_gradients(grads_and_vars=grads)

  train = get_train(t_loss, t_metrics)

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

def combine_metrics(log):
  metrics = {}
  for key in log[0]:
    values = []
    for entry in log:
      values.append(entry[key])
    metrics[key] = np.mean(values)
  return metrics

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
    epoch_metrics = []
    for batch in train_batches:
      tensors = [ train, update_global_step_t, t_metrics ]
      train_feed = {
          holds: batch['holds'],
          codes: batch['codes'],
          deltas: batch['deltas'],
          sequence_lens: batch['sequence_lens'],
          categories: batch['categories'],
          category_mask: train_mask,
          training: True,
          grad_clip: grad_clip_value if AUTO_CLIP else STATIC_CLIP,
        }
      _, _, metrics = sess.run(tensors,
          feed_dict=train_feed)

      step += 1
      epoch_metrics.append(metrics)

    metrics = combine_metrics(epoch_metrics)
    log_summary('train', metrics, step)

    grad_clip_value *= (1.0 - grad_clip_lambda)
    grad_clip_value += grad_clip_lambda * (grad_clip_mul * metrics['grad_norm'])

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
