import math
import tensorflow as tf

# Internal
import dataset

EMBED_WIDTH = 11
TIMES_WIDTH = 5

INPUT_DROPOUT = 0.0
RNN_INPUT_DROPOUT = 0.0
RNN_STATE_DROPOUT = 0.5
RNN_OUTPUT_DROPOUT = 0.0
RNN_USE_RESIDUAL = False
RNN_USE_BIDIR = False

DENSE_L2 = 0.001

GAUSSIAN_POOLING_VAR = 1.0
GAUSSIAN_POOLING_LEN_DELTA = 3.0

RNN_WIDTH = [ 32 ]
DENSE_POST_WIDTH = [ 32 ]
FEATURE_COUNT = 32

class Embedding():
  def __init__(self, name, max_code, width, regularizer=None):
    self.name = name
    self.width = width
    with tf.variable_scope(None, default_name=self.name):
      self.weights = tf.get_variable('weights', shape=(max_code, width),
                                     trainable=True,
                                     regularizer=regularizer)

  def apply(self, codes):
    with tf.name_scope(None, values=[ codes ], default_name=self.name):
      return tf.gather(self.weights, codes)

class Model():
  def __init__(self, training):
    self.l2 = tf.contrib.layers.l2_regularizer(DENSE_L2)
    self.training = training
    self.use_gaussian_pooling = True

    self.embedding = Embedding('embedding', dataset.MAX_CHAR + 2, EMBED_WIDTH)

    def create_rnn_cell(name):
      cells = []
      for i, width in enumerate(RNN_WIDTH):
        cell = tf.contrib.rnn.LSTMBlockCell(name='lstm_{}_{}'.format(name, i), \
            num_units=width)

        cell = tf.contrib.rnn.DropoutWrapper(cell,
            input_keep_prob=tf.where(training, 1.0 - RNN_INPUT_DROPOUT, 1.0),
            state_keep_prob=tf.where(training, 1.0 - RNN_STATE_DROPOUT, 1.0),
            output_keep_prob=tf.where(training, 1.0 - RNN_OUTPUT_DROPOUT, 1.0))

        if RNN_USE_RESIDUAL and i != 0:
          cell = tf.contrib.rnn.ResidualWrapper(cell)

        cells.append(cell)

      return tf.contrib.rnn.MultiRNNCell(cells)

    self.rnn_cell_fw = create_rnn_cell('fw')
    if RNN_USE_BIDIR:
      self.rnn_cell_bw = create_rnn_cell('bw')

    self.process_times = tf.layers.Dense(name='process_times',
                                         units=TIMES_WIDTH,
                                         kernel_regularizer=self.l2)

    self.post = []
    for i, width in enumerate(DENSE_POST_WIDTH):
      self.post.append(tf.layers.Dense(name='dense_post_{}'.format(i),
                                       units=width,
                                       activation=tf.nn.selu,
                                       kernel_regularizer=self.l2))

    self.features = tf.layers.Dense(name='features',
                                    units=FEATURE_COUNT,
                                    kernel_regularizer=self.l2)

  def apply_embedding(self, holds, codes, deltas):
    embedding = self.embedding.apply(codes)
    holds = tf.expand_dims(holds, axis=-1, name='expanded_holds')
    deltas = tf.expand_dims(deltas, axis=-1, name='expanded_deltas')

    times = tf.concat([ holds, deltas ], axis=-1, name='times')

    # Process holds+deltas
    times = self.process_times(times)

    series = tf.concat([ times, embedding ], axis=-1, name='full_input')
    series = tf.layers.dropout(series, rate=INPUT_DROPOUT,
        training=self.training)

    return series

  def build(self, holds, codes, deltas, sequence_len = None):
    batch_size = tf.shape(codes)[0]
    max_sequence_len = int(codes.shape[1])
    if sequence_len is None:
      sequence_len = tf.constant(max_sequence_len, dtype=tf.int32,
          shape=(1,))
      sequence_len = tf.tile(sequence_len, [ batch_size ])

    series = self.apply_embedding(holds, codes, deltas)
    frames = tf.unstack(series, axis=1, name='unstacked_output')

    if RNN_USE_BIDIR:
      outputs, _, _ = tf.nn.static_bidirectional_rnn(
          cell_fw=self.rnn_cell_fw,
          cell_bw=self.rnn_cell_bw,
          dtype=tf.float32,
          inputs=frames)
    else:
      outputs, _ = tf.nn.static_rnn(
          cell=self.rnn_cell_fw,
          dtype=tf.float32,
          inputs=frames)

    stacked_output = tf.stack(outputs, axis=1, name='stacked_output')

    # [ batch, sequence_len ]
    last_output_mask = tf.one_hot(sequence_len - 1, max_sequence_len,
        dtype=tf.float32)

    if self.use_gaussian_pooling:
      # [ 1, sequence_len ]
      indices = tf.expand_dims(tf.range(max_sequence_len), axis=0,
          name='sequence_indices')

      # [ batch, sequence_len ]
      mask = tf.cast(indices < tf.expand_dims(sequence_len, axis=-1),
          dtype=tf.float32,
          name='pre_mask')

      # [ batch ]
      len_delta = tf.random.uniform(
          (batch_size,),
          -GAUSSIAN_POOLING_LEN_DELTA,
          GAUSSIAN_POOLING_LEN_DELTA,
          name='len_delta')

      # [ batch, 1 ]
      random_len = tf.expand_dims(
          tf.cast(sequence_len, dtype=tf.float32) - 1.0 - len_delta,
          axis=-1,
          name='random_len')

      # [ batch, sequence_len ]
      gauss_x = (tf.cast(indices, dtype=tf.float32) - random_len) ** 2.0
      gauss_x /= 2.0 * (GAUSSIAN_POOLING_VAR ** 2)

      mask *= tf.exp(-gauss_x, name='gaussian_pre_mask')
      mask /= tf.reduce_sum(mask, axis=-1, keepdims=True,
          name='gaussian_mask_norm')

      mask = tf.where(self.training, mask, last_output_mask)
    else:
      mask = last_output_mask

    mask = tf.expand_dims(mask, axis=-1, name='last_mask')

    x = tf.reduce_sum(stacked_output * mask, axis=1,
        name='last_output')

    for post in self.post:
      x = post(x)

    x = self.features(x)

    return x

  def get_proxy_common(self, proxies, output, categories, category_count, \
      category_mask):
    positives = tf.gather(proxies, categories, axis=0,
        name='positive_proxies')

    negative_masks = tf.one_hot(categories, category_count, on_value=False,
        off_value=True, name='negative_mask')
    negative_masks = tf.logical_and(negative_masks, \
        tf.expand_dims(category_mask, axis=0))

    def apply_mask(mask):
      negatives = tf.boolean_mask(proxies, mask, axis=0, \
          name='batch_negatives')
      return negatives

    negatives = tf.map_fn(apply_mask, negative_masks, name='negatives',
        dtype=tf.float32)

    positive_distances = tf.norm(positives - output, axis=-1,
        name='positive_distances')
    negative_distances = tf.norm(negatives - tf.expand_dims(output, axis=1),
        axis=-1, name='negative_distances')

    metrics = {}
    for percentile in [ 5, 10, 25, 50, 75, 90, 95 ]:
      neg_p = tf.contrib.distributions.percentile(negative_distances,
          percentile, name='negative_{}'.format(percentile))
      metrics['negative_{}'.format(percentile)] = neg_p

      pos_p = tf.contrib.distributions.percentile(positive_distances,
          percentile, name='positive_{}'.format(percentile))
      metrics['positive_{}'.format(percentile)] = pos_p

    epsilon = 1e-12
    metrics['ratio_25'] = metrics['negative_25'] / \
        (metrics['positive_75'] + epsilon)
    metrics['ratio_10'] = metrics['negative_10'] / \
        (metrics['positive_90'] + epsilon)
    metrics['ratio_5'] = metrics['negative_5'] / \
        (metrics['positive_95'] + epsilon)

    return positive_distances, negative_distances, metrics


  # As in https://arxiv.org/pdf/1703.07464.pdf
  def get_proxy_loss(self, output, categories, category_count, \
      category_mask):
    with tf.name_scope('proxy_loss', [ output, categories, category_mask ]):
      proxies = tf.get_variable('points',
          trainable=True,
          initializer=tf.initializers.random_uniform(-1.0, 1.0),
          shape=(category_count, FEATURE_COUNT,))

      positive_distances, negative_distances, metrics = self.get_proxy_common( \
          proxies, output, categories, category_count, category_mask)

      exp_pos = tf.exp(-positive_distances, name='exp_pos')
      exp_neg = tf.exp(-negative_distances, name='exp_neg')

      total_exp_neg = tf.reduce_sum(exp_neg, axis=-1, name='total_exp_neg')

      epsilon = 1e-12
      ratio = exp_pos / (total_exp_neg + epsilon)

      loss = -tf.log(ratio + epsilon, name='loss_vector')
      loss = tf.reduce_mean(loss, name='loss')

      metrics['loss'] = loss

      return metrics

  def get_proxy_val_metrics(self, output, categories, category_count, \
      category_mask):
    with tf.name_scope('proxy_val_metrics', [ output, categories, \
        category_mask ]):
      # Compute proxies as mean points
      def compute_mean_proxy(category):
        points = tf.boolean_mask(output, tf.equal(categories, category),
            'category_points')
        return tf.reduce_mean(points, axis=0)

      proxies = tf.map_fn(compute_mean_proxy, tf.range(category_count),
          dtype=tf.float32)

      _, _, metrics = self.get_proxy_common(proxies, output, categories, \
          category_count, category_mask)

      return metrics
