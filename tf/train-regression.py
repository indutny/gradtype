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

# Maximum number of epochs to run for
MAX_EPOCHS = 500000

# Validate every epoch:
VALIDATE_EVERY = 5

# Learning rate
LR = 0.001

#
# Load dataset
#

train_dataset, validate_dataset = dataset.load(mode='regression')
train_dataset = dataset.flatten_dataset(train_dataset)
validate_dataset = dataset.flatten_dataset(validate_dataset)

#
# Initialize model
#

input_shape = (None, dataset.MAX_SEQUENCE_LEN,)

codes = tf.placeholder(tf.int32, shape=input_shape, name='codes')
deltas = tf.placeholder(tf.float32, shape=input_shape, name='deltas')
categories = tf.placeholder(tf.int32, shape=(None,), name='categories')
training = tf.placeholder(tf.bool, shape=(), name='training')

model = Model(training=training)

output = model.build(codes, deltas)
t_metrics = model.get_regression_metrics(output, categories)

#
# Initialize optimizer
#

optimizer = tf.train.MomentumOptimizer(LR, momentum=0.5)
t_reg_loss = tf.losses.get_regularization_loss()
train = optimizer.minimize(t_metrics['loss'] + t_reg_loss)

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

  step = 0
  for epoch in range(0, MAX_EPOCHS):
    train_batches = dataset.gen_regression(train_dataset)
    validate_batches = dataset.gen_regression(validate_dataset)

    saver.save(sess, LOG_DIR, global_step=step)
    print('Epoch {}'.format(epoch))
    for batch in train_batches:
      tensors = [ train, t_metrics, t_reg_loss ]
      _, metrics, reg_loss = sess.run(tensors, feed_dict={
        codes: batch['codes'],
        deltas: batch['deltas'],
        categories: batch['categories'],
        training: True,
      })
      metrics['regularization_loss'] = reg_loss
      log_summary('train', metrics, step)

      step += 1

    print('Validation...')
    mean_metrics = None
    for batch in validate_batches:
      metrics = sess.run(t_metrics, feed_dict={
        codes: batch['codes'],
        deltas: batch['deltas'],
        categories: batch['categories'],
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
