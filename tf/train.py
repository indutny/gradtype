import tensorflow as tf

# Internal
import dataset
from model import Model

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
loss = model.compute_loss(output, category_count)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for epoch in range(0, MAX_EPOCHS):
    train_batches = dataset.gen_batches(train)
    if epoch % VALIDATE_EVERY == 0:
      validate_batches = dataset.gen_batches(validate, batch_size=BATCH_SIZE)
    else:
      validate_batches = None

    print('Epoch {}'.format(epoch))
    for batch in train_batches:
      # Run model
      print(sess.run(loss, feed_dict={
        codes: batch['codes'],
        deltas: batch['deltas'],
        category_count: len(train),
      }))
