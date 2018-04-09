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

model = Model()

with tf.Session() as sess:
  for epoch in range(0, MAX_EPOCHS):
    train_batches = dataset.gen_batches(train)
    if epoch % VALIDATE_EVERY == 0:
      validate_batches = dataset.gen_batches(validate, batch_size=BATCH_SIZE)
    else:
      validate_batches = None

    for batch in train_batches:
      output = model(batch['codes'], batch['deltas'])

      # Initialize global variables after building model
      sess.run(tf.global_variables_initializer())

      # Run model
      loss = model.compute_loss(output, BATCH_SIZE)
      print(sess.run(loss))
      exit(0)
