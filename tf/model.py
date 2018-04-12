import tensorflow as tf

# Internal
import dataset
from gru import GRUCell

EMBED_WIDTH = 7
DENSE_PRE_COUNT = 1
DENSE_PRE_WIDTH = 32
DENSE_PRE_RESIDUAL_COUNT = 3
GRU_WIDTH = 128
DENSE_POST_WIDTH = [ 256, 128 ]
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
  def __init__(self, is_training):
    self.l2 = tf.contrib.layers.l2_regularizer(0.001)

    self.embedding = Embedding('embedding', dataset.MAX_CHAR + 2, EMBED_WIDTH)

    self.pre = []
    for i in range(0, DENSE_PRE_COUNT):
      self.pre.append(tf.layers.Dense(name='dense_pre_{}'.format(i),
                                      units=DENSE_PRE_WIDTH,
                                      activation=tf.nn.selu,
                                      kernel_regularizer=self.l2))

    self.pre_residual = []
    for i in range(0, DENSE_PRE_RESIDUAL_COUNT):
      self.pre_residual.append([
          tf.layers.Dense(name='dense_pre_residual_minor_{}'.format(i),
                          units=int(DENSE_PRE_WIDTH / 2),
                          activation=tf.nn.selu,
                          kernel_regularizer=self.l2),
          tf.layers.Dense(name='dense_pre_residual_major_{}'.format(i),
                          units=DENSE_PRE_WIDTH,
                          kernel_regularizer=self.l2) ])

    self.gru = GRUCell(name='gru', units=GRU_WIDTH, is_training=is_training)

    self.post = []
    for width in DENSE_POST_WIDTH:
      self.post.append(tf.layers.Dense(name='dense_post_{}'.format(i),
                                       units=width,
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

      for [ minor, major ] in self.pre_residual:
        frame = tf.nn.selu(frame + major(minor(frame)))

      x, state = self.gru(frame, state)

    for post in self.post:
      x = post(x)

    x = self.features(x)

    return x

  # Batch Hard as in https://arxiv.org/pdf/1703.07737.pdf
  def get_metrics(self, output, category_count, batch_size,
                  margin=0.2, epsilon=1e-18, loss_kind='triplet'):
    with tf.name_scope('metrics', [ output ]):
      margin = tf.constant(margin, dtype=tf.float32, name='margin')
      epsilon = tf.constant(epsilon, dtype=tf.float32, name='epsilon')
      zero = tf.constant(0.0, dtype=tf.float32, name='zero')

      categories = tf.expand_dims(tf.range(category_count), axis=-1,
          name='categories')
      categories = tf.tile(categories, [ 1, batch_size ],
          name='tiled_categories')

      row_count = category_count * batch_size
      categories = tf.reshape(categories, (row_count,), 'reshaped_categories')

      # same_mask.shape =
      #  (category_count * batch_size, category_count * batch_size)
      same_mask = tf.equal(tf.expand_dims(categories, axis=0),
          tf.expand_dims(categories, axis=1),
          name='same_mask_with_dups')
      not_same_mask = tf.logical_not(same_mask, name='not_same_mask')
      same_mask = tf.logical_xor(same_mask, tf.eye(row_count, dtype=tf.bool),
          name='same_mask')

      # Compute all-to-all euclidian distances
      distances = tf.expand_dims(output, axis=0) - \
          tf.expand_dims(output, axis=1)
      # distances.shape = same_mask.shape
      distances2 = tf.reduce_sum(distances ** 2, axis=-1, name='distances2')
      distances = tf.sqrt(distances2 + epsilon, name='distances')

      if loss_kind == 'batch_hard':
        positive_mask = tf.cast(same_mask, tf.float32, 'positive_mask')
        negative_mask = tf.cast(not_same_mask, tf.float32, 'negative_mask')

        positive_distances = distances * positive_mask
        negative_distances = distances * negative_mask + \
            (1 - negative_mask) * 1e9

        hard_positives = tf.reduce_max(positive_distances, axis=-1,
            name='hard_positives')
        hard_negatives = tf.reduce_min(negative_distances, axis=-1,
            name='hard_negatives')

        triplet_distance = hard_positives - hard_negatives
      elif loss_kind == 'triplet':
        def compute_soft_negative(t):
          positive = t[0]
          negatives = t[1]

          inf = tf.tile([ float('inf') ], tf.shape(negatives))
          soft_negatives = tf.where(negatives > positive, negatives, inf,
                                    name='soft_negatives')
          soft_negative = tf.gather(negatives, tf.argmin(soft_negatives))
          return positive - soft_negative

        def compute_triplet_row(t):
          # Positive distances between anchor and all positives
          positives = tf.boolean_mask(t[0], t[1], name='triplet_positives')
          # Negative distances between anchor and all negatives
          negatives = tf.boolean_mask(t[0], t[2], name='triplet_negatives')

          negatives = tf.expand_dims(negatives, axis=0)
          negatives = tf.tile(negatives, [ tf.shape(positives)[0], 1 ])

          # For each anchor-positive - find soft negative
          return tf.map_fn(compute_soft_negative, (positives, negatives),
              dtype=tf.float32)

        triplet_distance = tf.map_fn(compute_triplet_row,
            (distances, same_mask, not_same_mask),
            dtype=tf.float32)
      else:
        raise Exception('Unknown loss kind "{}"'.format(loss_kind))

      triplet_distance += margin
      loss = tf.maximum(triplet_distance, zero)
      loss = tf.reduce_mean(loss, name='loss')

      metrics = {}
      metrics['loss'] = loss
      metrics['mean_positive'] = \
          tf.reduce_mean(tf.boolean_mask(distances, same_mask),
                         name='mean_positive')
      metrics['mean_negative'] = \
          tf.reduce_mean(tf.boolean_mask(distances, not_same_mask),
                         name='mean_negative')
      metrics['active_triplets'] = \
          tf.reduce_mean(tf.cast(tf.greater(triplet_distance, zero), \
                                 tf.float32),
                         name='active_triplets')
      metrics['l2_norm'] = tf.reduce_mean(distances2, name='l2_norm')

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
