import math
import tensorflow as tf

# Internal
import dataset

EMBED_WIDTH = 11
TIMES_WIDTH = 5

INPUT_DROPOUT = 0.0

RADIUS_MAX_STEP = 20000.0

DENSE_L2 = 0.001

GAUSSIAN_POOLING_VAR = 1.0
GAUSSIAN_POOLING_LEN_DELTA = 3.0

RNN_WIDTH = 32
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
    self.use_gaussian_pooling = False

    self.use_cosine = True

    self.use_lcml = True
    self.margin = 0.2
    self.radius = 13.331313782506344

    self.embedding = Embedding('embedding', dataset.MAX_CHAR + 2, EMBED_WIDTH)

    self.rnn_cell = tf.contrib.rnn.LSTMBlockCell(name='lstm_cell',
        num_units=RNN_WIDTH)

    self.input_dropout = tf.keras.layers.Dropout(name='input_dropout',
        rate=INPUT_DROPOUT)

    self.process_times = tf.layers.Dense(name='process_times',
                                         units=TIMES_WIDTH,
                                         kernel_regularizer=self.l2)

    self.post = []
    self.post_bn = []
    for i, width in enumerate(DENSE_POST_WIDTH):
      self.post.append(tf.layers.Dense(name='dense_post_{}'.format(i),
                                       units=width,
                                       activation=tf.nn.relu,
                                       kernel_regularizer=self.l2))
      self.post_bn.append(tf.keras.layers.BatchNormalization(
          name='bn_post_{}'.format(i)))

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
    series = self.input_dropout(series, training=self.training)

    return series

  def build(self, holds, codes, deltas, sequence_len = None):
    batch_size = tf.shape(codes)[0]
    max_sequence_len = int(codes.shape[1])
    if sequence_len is None:
      sequence_len = tf.constant(max_sequence_len, dtype=tf.int32,
          shape=(1,))
      sequence_len = tf.tile(sequence_len, [ batch_size ])

    series = self.apply_embedding(holds, codes, deltas)
    series = tf.unstack(series, axis=1, name='unstacked_series')

    outputs, _ = tf.nn.static_rnn(
          cell=self.rnn_cell,
          dtype=tf.float32,
          inputs=series)
    outputs = tf.stack(outputs, axis=1, name='stacked_outputs')

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

    x = tf.reduce_sum(outputs * mask, axis=1,
        name='last_output')

    for (post, bn) in zip(self.post, self.post_bn):
      x = post(x)
      x = bn(x, training=self.training)

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

    if self.use_cosine:
      def cosine(a, b):
        dot = tf.reduce_sum(a * b, axis=-1)
        a_norm = tf.norm(a, axis=-1)
        b_norm = tf.norm(b, axis=-1)
        cos = dot / a_norm / b_norm
        return 1.0 - cos

      positive_distances = cosine(positives, output)
      negative_distances = cosine(negatives, tf.expand_dims(output, axis=1))
    else:
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
  # TODO(indutny) Consider: http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_CosFace_Large_Margin_CVPR_2018_paper.pdf
  def get_proxy_loss(self, output, categories, category_count, \
      category_mask, step):
    with tf.name_scope('proxy_loss', [ output, categories, category_mask ]):
      proxies_init = tf.initializers.random_uniform(-1.0, 1.0)( \
          (category_count, FEATURE_COUNT,))
      proxies_init = tf.math.l2_normalize(proxies_init, axis=-1,
          name='sphere_initial_proxies')
      proxies = tf.get_variable('points',
          trainable=True,
          initializer=proxies_init)

      positive_distances, negative_distances, metrics = self.get_proxy_common( \
          proxies, output, categories, category_count, category_mask)

      epsilon = 1e-12

      radius = 1.0 + (self.radius - 1.0) * tf.minimum(1.0,
          tf.cast(step, dtype=tf.float32) / RADIUS_MAX_STEP)

      if self.use_lcml:
        exp_pos = tf.exp(-radius * (positive_distances + self.margin),
            name='exp_pos')
        exp_neg = tf.exp(-radius * negative_distances, name='exp_neg')

        total_exp_neg = tf.reduce_sum(exp_neg, axis=-1, name='total_exp_neg')

        ratio = exp_pos / (exp_pos + total_exp_neg + epsilon)
      else:
        exp_pos = tf.exp(-positive_distances, name='exp_pos')
        exp_neg = tf.exp(-negative_distances, name='exp_neg')

        total_exp_neg = tf.reduce_sum(exp_neg, axis=-1, name='total_exp_neg')

        ratio = exp_pos / (total_exp_neg + epsilon)

      loss = -tf.log(ratio + epsilon, name='loss_vector')
      loss = tf.reduce_mean(loss, name='loss')

      metrics['loss'] = loss
      metrics['radius'] = radius

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
