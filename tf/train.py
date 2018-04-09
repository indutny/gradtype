import os
import time

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

BATCH_SIZE = 32

train, validate = dataset.load()

model = Model(BATCH_SIZE)

input_shape = (None, dataset.MAX_SEQUENCE_LEN,)

codes = tf.placeholder(tf.int32, shape=input_shape, name='codes')
deltas = tf.placeholder(tf.float32, shape=input_shape, name='deltas')
category_count = tf.placeholder(tf.int32, (), name='category_count')

output = model.build(codes, deltas, category_count)
metrics = model.get_metrics(output, category_count)

writer = tf.summary.FileWriter(LOG_DIR)
writer.add_graph(tf.get_default_graph())

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  with tf.name_scope('train'):
    tf.summary.scalar('loss', loss)

  merged_summary = tf.summary.merge_all()

  step = 0
  for epoch in range(0, MAX_EPOCHS):
    train_batches = dataset.gen_batches(train)
    if epoch % VALIDATE_EVERY == 0:
      validate_batches = dataset.gen_batches(validate, batch_size=BATCH_SIZE)
    else:
      validate_batches = None

    print('Epoch {}'.format(epoch))
    for batch in train_batches:
      # Run model
      summary = sess.run(metrics, feed_dict={
        codes: batch['codes'],
        deltas: batch['deltas'],
        category_count: len(train),
      })
      writer.add_summary(summary, step)
      step += 1
