import math
import tensorflow as tf

# Internal
import dataset

EMBED_WIDTH = 16
TIMES_WIDTH = 16

INPUT_DROPOUT = 0.0
POST_RNN_DROPOUT = 0.0

DENSE_L2 = 0.0

RNN_WIDTH = [ 16, 16 ]
DENSE_POST_WIDTH = [ (128, 0.0) ]
FEATURE_COUNT = 32

ANNEAL_MAX_STEP = 10000.0
MAX_SPHERE_STRENGTH = 0.9

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
    self.use_sphereface = False
    self.use_arcface = True
    self.arcface_m1 = 1.35 # cos(m1 * x + m2) - m3
    self.arcface_m2 = 0.0
    self.arcface_m3 = 0.0
    self.anneal_distances = False

    self.radius = 9.2

    self.embedding = Embedding('embedding', dataset.MAX_CHAR + 2, EMBED_WIDTH)

    self.rnn_cells = [
        tf.contrib.rnn.LSTMBlockCell(name='lstm_cell_{}'.format(i),
          num_units=width)
        for i, width in enumerate(RNN_WIDTH)
    ]

    self.input_dropout = tf.keras.layers.GaussianDropout(name='input_dropout',
        rate=INPUT_DROPOUT)
    self.post_rnn_dropout = tf.keras.layers.Dropout(
        name='post_rnn_dropout',
        rate=POST_RNN_DROPOUT)

    self.process_times = tf.layers.Dense(name='process_times',
                                         units=TIMES_WIDTH,
                                         activation=tf.nn.relu,
                                         kernel_regularizer=self.l2)

    self.post = []
    for i, (width, dropout) in enumerate(DENSE_POST_WIDTH):
      dense = tf.layers.Dense(name='dense_post_{}'.format(i),
                              units=width,
                              activation=tf.nn.relu,
                              kernel_regularizer=self.l2)
      dropout = tf.keras.layers.Dropout(
          name='dropout_post_{}'.format(i),
          rate=dropout)
      self.post.append({ 'dense': dense, 'dropout': dropout })

    self.features = tf.layers.Dense(name='features',
                                    units=FEATURE_COUNT,
                                    kernel_regularizer=self.l2)

  def apply_embedding(self, holds, codes, deltas):
    embedding = self.embedding.apply(codes)
    holds = tf.expand_dims(holds, axis=-1, name='expanded_holds')
    deltas = tf.expand_dims(deltas, axis=-1, name='expanded_deltas')

    times = tf.concat([ holds, deltas ], axis=-1, name='times')
    times = self.input_dropout(times, training=self.training)

    # Process holds+deltas
    times = self.process_times(times)

    series = tf.concat([ times, embedding ], axis=-1, name='full_input')

    return series, embedding

  def build(self, holds, codes, deltas, sequence_len=None):
    batch_size = tf.shape(codes)[0]
    max_sequence_len = int(codes.shape[1])
    if sequence_len is None:
      sequence_len = tf.constant(max_sequence_len, dtype=tf.int32,
          shape=(1,))
      sequence_len = tf.tile(sequence_len, [ batch_size ])

    series, embedding = self.apply_embedding(holds, codes, deltas)
    series = tf.unstack(series, axis=1, name='unstacked_series')

    for cell in self.rnn_cells:
      series, _ = tf.nn.static_rnn(
            cell=cell,
            dtype=tf.float32,
            inputs=series)

    seq_index = tf.expand_dims(tf.range(max_sequence_len), axis=0,
        name='seq_index')
    mask = seq_index < tf.expand_dims(sequence_len, axis=-1)
    mask = tf.cast(mask, dtype=tf.float32, name='mask')
    mask /= tf.reduce_sum(mask, axis=-1, keepdims=True)
    mask = tf.expand_dims(mask, axis=-1, name='avg_mask')

    outputs = tf.stack(series, axis=1, name='stacked_outputs')
    x = outputs * mask
    x = tf.reduce_sum(x, axis=1)
    x = self.post_rnn_dropout(x, training=self.training)

    for entry in self.post:
      x = entry['dense'](x)
      x = entry['dropout'](x, training=self.training)

    x = self.features(x)
    x = tf.math.l2_normalize(x, axis=-1)

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

    def cosine(normed_target, normed_features):
      cos = tf.reduce_sum(normed_target * normed_features, axis=-1)
      dist = 1.0 - cos
      return cos, dist

    positive_distances, positive_metrics = cosine(positives, output)
    negative_distances, negative_metrics = cosine(negatives, \
        tf.expand_dims(output, axis=1))

    metrics = {}
    for percentile in [ 1, 5, 10, 25, 50, 75, 90, 95, 99 ]:
      neg_p = tf.contrib.distributions.percentile(negative_metrics,
          percentile, name='negative_{}'.format(percentile))
      metrics['negative_{}'.format(percentile)] = neg_p

      pos_p = tf.contrib.distributions.percentile(positive_metrics,
          percentile, name='positive_{}'.format(percentile))
      metrics['positive_{}'.format(percentile)] = pos_p

    epsilon = 1e-12
    metrics['ratio_25'] = metrics['negative_25'] / \
        (metrics['positive_75'] + epsilon)
    metrics['ratio_10'] = metrics['negative_10'] / \
        (metrics['positive_90'] + epsilon)
    metrics['ratio_5'] = metrics['negative_5'] / \
        (metrics['positive_95'] + epsilon)
    metrics['ratio_1'] = metrics['negative_1'] / \
        (metrics['positive_99'] + epsilon)

    return positive_distances, negative_distances, metrics


  # As in https://arxiv.org/pdf/1703.07464.pdf
  # More like in: http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_CosFace_Large_Margin_CVPR_2018_paper.pdf
  # TODO(indutny): try http://ydwen.github.io/papers/WenECCV16.pdf
  # TODO(indutny): try http://openaccess.thecvf.com/content_cvpr_2018/papers/Zheng_Ring_Loss_Convex_CVPR_2018_paper.pdf
  # TODO(indutny): try https://arxiv.org/pdf/1704.08063.pdf
  # TODO(indutny): try https://arxiv.org/pdf/1703.09507.pdf
  # See http://proceedings.mlr.press/v48/liud16.pdf
  # TODO(indutny): https://arxiv.org/pdf/1801.07698.pdf
  def get_proxy_loss(self, output, categories, category_count, \
      category_mask, step):
    with tf.name_scope('proxy_loss', [ output, categories, category_mask ]):
      proxies_init = tf.initializers.random_uniform(-1.0, 1.0)( \
          (category_count, FEATURE_COUNT,))
      proxies = tf.get_variable('points',
          trainable=True,
          initializer=proxies_init)
      proxies = tf.math.l2_normalize(proxies, axis=-1,
          name='normalized_proxies')

      positive_distances, negative_distances, _ = self.get_proxy_common( \
          proxies, output, categories, category_count, category_mask)

      # NOTE: We use same mean proxies for the metrics as in validation

      mean_proxies = self.mean_proxies(output, categories, category_count)
      _, _, metrics = self.get_proxy_common( \
          mean_proxies, output, categories, category_count, category_mask)

      epsilon = 1e-23

      # Primitive spline with derivatives equal to zero at both start and
      # max step
      anneal_lambda = tf.clip_by_value(
          tf.cast(step, dtype=tf.float32) / ANNEAL_MAX_STEP,
          0.0,
          1.0)
      anneal_lambda = -2.0 * (anneal_lambda ** 3.0) + \
          3.0 * (anneal_lambda ** 2.0)

      radius = 1.0 + (self.radius - 1.0) * anneal_lambda

      metrics['anneal_lambda'] = anneal_lambda
      metrics['radius'] = radius

      if self.use_sphereface:
        # cos(2x) = 2.0 * cos^2(x) - 1
        double_cos = 2.0 * (positive_distances ** 2.0) - 1.0
        k = tf.cast(positive_distances <= 0.0, dtype=tf.float32)
        sign = (-1.0) ** k

        psi = sign * double_cos - 2 * k

        # Regardless of annealing, do not cross maximum strengh of double cos
        # It leads to gradient collapse, and possibly worse learning than
        # it should be.
        psi *= MAX_SPHERE_STRENGTH
        psi += (1.0 - MAX_SPHERE_STRENGTH) * positive_distances

        # TODO(indutny): try annealing again
        if self.anneal_distances:
          positive_distances *= (1.0 - anneal_lambda)
          positive_distances += anneal_lambda * psi
        else:
          positive_distances = psi
      elif self.use_arcface:
        psi = tf.math.acos(positive_distances)
        # cos(m1 * x + m2) - m3
        psi *= self.arcface_m1
        psi += self.arcface_m2
        psi = tf.math.cos(psi)
        psi -= self.arcface_m3

        # according to the paper - no annealing is necessary
        positive_distances = psi
      else:
        # Just apply margin
        positive_distances -= self.arcface_m3

      positive_distances *= radius
      negative_distances *= radius

      exp_pos = tf.exp(positive_distances, name='exp_pos')
      exp_neg = tf.exp(negative_distances, name='exp_neg')

      total_exp_neg = tf.reduce_sum(exp_neg, axis=-1, name='total_exp_neg')

      ratio = exp_pos / (exp_pos + total_exp_neg + epsilon)

      loss = -tf.log(ratio + epsilon, name='loss_vector')
      loss = tf.reduce_mean(loss, name='loss')

      metrics['loss'] = loss

      return metrics

  def get_proxy_val_metrics(self, output, categories, category_count, \
      category_mask):
    with tf.name_scope('proxy_val_metrics', [ output, categories, \
        category_mask ]):
      proxies = self.mean_proxies(output, categories, category_count)
      _, _, metrics = self.get_proxy_common(proxies, output, categories, \
          category_count, category_mask)

      return metrics

  def mean_proxies(self, output, categories, category_count):
    # Compute proxies as mean points
    def compute_mean_proxy(category):
      points = tf.boolean_mask(output, tf.equal(categories, category),
          'category_points')
      return tf.reduce_mean(points, axis=0)

    result = tf.map_fn(compute_mean_proxy, tf.range(category_count),
        dtype=tf.float32)
    result = tf.math.l2_normalize(result, axis=-1)
    return result
