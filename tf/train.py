import tensorflow as tf

# Internal
import dataset
from model import Model

train, validate = dataset.load()

model = Model()

with tf.Session() as sess:
  codes = tf.constant([ [ 1, 2, 3 ] ], dtype=tf.int32)
  deltas = tf.constant([ [ 0.1, 0.2, 0.3 ] ], dtype=tf.float32)

  concat = model(codes, deltas)

  # Initialize global variables after building model
  sess.run(tf.global_variables_initializer())

  print(sess.run(concat))
