import tensorflow as tf

# Internal
import dataset

EMBED_WIDTH = 7
DENSE_PRE_COUNT = 3
DENSE_PRE_WIDTH = 32
GRU_WIDTH = 128
DENSE_POST_COUNT = 1
DENSE_POST_WIDTH = 128
FEATURE_COUNT = 128

class Embedding():
  def __init__(self, name, max_code, width):
    self.width = width
    with tf.variable_scope('layer/' + name):
      self.weights = tf.get_variable('weights', (max_code, width))

  def apply(self, codes):
    return tf.gather(self.weights, codes)

class Model():
  def __init__(self):
    self.l2 = tf.contrib.layers.l2_regularizer(0.001)

    self.embedding = Embedding('embedding', dataset.MAX_CHAR + 2, EMBED_WIDTH)

    self.pre = []
    for i in range(0, DENSE_PRE_COUNT):
      self.pre.append(tf.layers.Dense(name='dense_pre_{}'.format(i),
                                      units=DENSE_PRE_WIDTH,
                                      activation=tf.nn.selu,
                                      kernel_regularizer=self.l2))

    self.gru = tf.contrib.rnn.GRUCell(name='gru', num_units=GRU_WIDTH)
    self.gru_regularized = False

    self.features = tf.layers.Dense(name='features',
                                    units=FEATURE_COUNT,
                                    kernel_regularizer=self.l2)

    self.post = []
    for i in range(0, DENSE_POST_COUNT):
      self.post.append(tf.layers.Dense(name='dense_post_{}'.format(i),
                                       units=DENSE_POST_WIDTH,
                                       activation=tf.nn.selu,
                                       kernel_regularizer=self.l2))

  def __call__(self, codes, deltas):
    batch_size = codes.shape[0]
    sequence_len = codes.shape[1]

    embedding = self.embedding.apply(codes)
    deltas = tf.expand_dims(deltas, axis=-1)
    series = tf.concat([ deltas, embedding ], axis=-1)

    state = self.gru.zero_state(batch_size, tf.float32)
    x = None
    for i in range(0, sequence_len):
      frame = series[:, i]
      for pre in self.pre:
        frame = pre(frame)

      state, x = self.gru(frame, state)

    for post in self.post:
      x = post(x)

    x = self.features(x)

    # Regularize GRU
    # TODO(indutny): just use custom GRU
    if not self.gru_regularized:
      self.gru_regularized = True

      gru_kernels = [
          self.gru.trainable_weights[0], self.gru.trainable_weights[2] ]
      for kernel in gru_kernels:
        self.gru.add_loss(self.l2(kernel))

    return x
