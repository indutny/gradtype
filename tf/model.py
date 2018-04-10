import tensorflow as tf

# Internal
import dataset
from gru import GRUCell

EMBED_WIDTH = 7
DENSE_PRE_COUNT = 3
DENSE_PRE_WIDTH = 32
GRU_WIDTH = 128
DENSE_POST_COUNT = 1
DENSE_POST_WIDTH = 128
FEATURE_COUNT = 128

class Embedding():
  def __init__(self, name, max_code, width):
    self.name = name
    self.width = width
    with tf.variable_scope(None, default_name=self.name):
      self.weights = tf.get_variable('weights', shape=(max_code, width))

  def apply(self, codes):
    with tf.name_scope(None, values=[ codes ], default_name=self.name):
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

    self.gru = GRUCell(name='gru', units=GRU_WIDTH)

    self.post = []
    for i in range(0, DENSE_POST_COUNT):
      self.post.append(tf.layers.Dense(name='dense_post_{}'.format(i),
                                       units=DENSE_POST_WIDTH,
                                       activation=tf.nn.selu,
                                       kernel_regularizer=self.l2))

    self.features = tf.layers.Dense(name='features',
                                    units=FEATURE_COUNT,
                                    kernel_regularizer=self.l2)

  def build(self, codes, deltas):
    sequence_len = int(codes.shape[1])

    embedding = self.embedding.apply(codes)
    deltas = tf.expand_dims(deltas, axis=-1)
    series = tf.concat([ deltas, embedding ], axis=-1)

    state = self.gru.build((None, 32))
    x = None
    for i in range(0, sequence_len):
      frame = series[:, i]
      for pre in self.pre:
        frame = pre(frame)

      x, state = self.gru(frame, state)

    for post in self.post:
      x = post(x)

    x = self.features(x)

    return x

  # Batch Hard as in https://arxiv.org/pdf/1703.07737.pdf
  def get_metrics(self, output, category_count, batch_size,
                  margin=0.2, epsilon=1e-18):
    with tf.name_scope('metrics', [ output ]):
      margin = tf.constant(margin, dtype=tf.float32)
      epsilon = tf.constant(epsilon, dtype=tf.float32)
      zero = tf.constant(0.0, dtype=tf.float32)

      categories = tf.expand_dims(tf.range(category_count), axis=-1)
      categories = tf.tile(categories, [ 1, batch_size ])

      row_count = category_count * batch_size
      categories = tf.reshape(categories, (row_count,))

      # same_mask.shape =
      #  (category_count * batch_size, category_count * batch_size)
      same_mask = tf.equal(tf.expand_dims(categories, axis=0),
          tf.expand_dims(categories, axis=1))
      not_same_mask = tf.logical_not(same_mask)
      same_mask = tf.logical_xor(same_mask, tf.eye(row_count, dtype=tf.bool))

      # Compute all-to-all euclidian distances
      distances = tf.expand_dims(output, axis=0) - \
          tf.expand_dims(output, axis=1)
      # distances.shape = same_mask.shape
      distances2 = tf.reduce_sum(distances ** 2, axis=-1)
      distances = tf.sqrt(distances2 + epsilon)

      positive_mask = tf.cast(same_mask, tf.float32)
      negative_mask = tf.cast(not_same_mask, tf.float32)

      positive_distances = distances * positive_mask
      negative_distances = distances * negative_mask + (1 - negative_mask) * 1e9

      hard_positives = tf.reduce_max(positive_distances, axis=-1)
      hard_negatives = tf.reduce_min(negative_distances, axis=-1)

      triplet_distance = hard_positives - hard_negatives

      loss = tf.nn.softplus(triplet_distance)
      loss = tf.reduce_mean(loss, axis=-1)

      metrics = {}
      metrics['loss'] = loss
      metrics['mean_positive'] = \
          tf.reduce_mean(tf.boolean_mask(distances, same_mask))
      metrics['mean_negative'] = \
          tf.reduce_mean(tf.boolean_mask(distances, not_same_mask))
      metrics['active_triplets'] = \
          tf.reduce_mean(tf.cast(tf.greater(triplet_distance, zero), \
              tf.float32))
      metrics['l2_norm'] = tf.reduce_mean(distances2)

      return metrics

  def get_regression_metrics(self, output, categories):
    with tf.name_scope('regression_loss', [ output, categories ]):
      categories_one_hot = tf.one_hot(categories, output.shape[1], axis=-1)
      loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, \
          labels=categories_one_hot)
      loss = tf.reduce_mean(loss)
      accuracy = tf.equal(tf.cast(tf.argmax(output, axis=-1), tf.int32),
                          tf.cast(categories, tf.int32))
      accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
      return { 'loss': loss, 'accuracy': accuracy }
