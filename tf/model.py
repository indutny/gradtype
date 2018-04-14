import tensorflow as tf

# Internal
import dataset
from gru import GRUCell

EMBED_WIDTH = 7
DENSE_PRE_COUNT = 1
DENSE_PRE_WIDTH = 32
DENSE_PRE_RESIDUAL_COUNT = 3

CONV_FILTERS = DENSE_PRE_WIDTH
CONV_KERNEL = 8
CONV_COUNT = 0

GRU_WIDTH = [ 128 ]
DENSE_POST_WIDTH = [ 128 ]
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
  def __init__(self, training):
    self.l2 = tf.contrib.layers.l2_regularizer(0.001)
    self.training = training
    self.use_pooling = True

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

    self.conv = []
    for i in range(0, CONV_COUNT):
      self.conv.append(tf.layers.Conv1D(name='conv_{}'.format(i),
                                        filters=CONV_FILTERS,
                                        kernel_size=CONV_KERNEL,
                                        padding='same',
                                        kernel_regularizer=self.l2,
                                        activation=tf.nn.selu))

    self.gru = []
    for i, width in enumerate(GRU_WIDTH):
      self.gru.append(GRUCell(name='gru_{}'.format(i), units=width,
                              training=training))

    self.gru_dropouts = []
    for i in range(0, len(GRU_WIDTH)):
      self.gru_dropouts.append(tf.layers.Dropout(name='dropout_{}'.format(i),
                                                 rate=0.5))

    self.post = []
    for i, width in enumerate(DENSE_POST_WIDTH):
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

    states = []
    for input_width, gru in zip([ DENSE_PRE_WIDTH ] + GRU_WIDTH[:-1], self.gru):
      states.append(gru.build((None, input_width)))

    if len(self.conv) != 0:
      conv_series = series
      for conv in self.conv:
        conv_series = conv(conv_series)

    gru_outputs = []
    for i in range(0, sequence_len):
      frame = series[:, i]

      if len(self.conv) != 0:
        conv_frame = conv_series[:, i]
        frame = tf.concat([ frame, conv_frame ], axis=-1)

      for pre in self.pre:
        frame = pre(frame)

      for [ minor, major ] in self.pre_residual:
        frame = tf.nn.selu(frame + major(minor(frame)))

      next_states = []
      for state, gru, drop in zip(states, self.gru, self.gru_dropouts):
        frame, state = gru(frame, state)
        if drop != None:
          frame = drop.apply(frame, training=self.training)
        next_states.append(state)
      states = next_states

      gru_outputs.append(frame)

    if self.use_pooling:
      x = tf.layers.max_pooling1d(gru_outputs, (sequence_len), strides=1)
    else:
      x = gru_outputs[-1]

    for post in self.post:
      x = post(x)

    x = self.features(x)

    return x

  def build_conv(self, codes, deltas):
    sequence_len = int(codes.shape[1])

    embedding = self.embedding.apply(codes)
    deltas = tf.expand_dims(deltas, axis=-1)
    series = tf.concat([ deltas, embedding ], axis=-1)

    series = tf.layers.conv1d(series, 128, 8, kernel_regularizer=self.l2,
                              activation=tf.nn.selu)
    series = tf.layers.batch_normalization(series)
    series = tf.layers.conv1d(series, 256, 8, kernel_regularizer=self.l2,
                              activation=tf.nn.selu)
    series = tf.layers.batch_normalization(series)
    series = tf.layers.conv1d(series, 128, 8, kernel_regularizer=self.l2,
                              activation=tf.nn.selu)
    series = tf.layers.batch_normalization(series)
    pool = tf.layers.average_pooling1d(series, (sequence_len - 21), strides=1)

    x = tf.reshape(pool, shape=(tf.shape(pool)[0], 128,))
    return self.features(x)

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

          is_soft = tf.greater(negatives, positive, name='is_soft')
          is_hard = tf.logical_not(is_soft, name='is_hard')

          soft_negatives = tf.boolean_mask(negatives, mask=is_soft,
              name='soft_negatives')
          hard_negatives = tf.boolean_mask(negatives, mask=is_hard,
              name='hard_negatives')

          soft_negative = tf.reduce_min(soft_negatives, name='soft_negative')
          soft_hard_negative = tf.reduce_max(hard_negatives,
              name='soft_hard_negative')

          selection = tf.where(tf.shape(soft_negatives)[-1] == 0,
              soft_hard_negative,
              soft_negative, name='selected_soft_negative')
          return positive - selection

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
