import math
import tensorflow as tf

# Internal
import dataset

EMBED_WIDTH = 16
TIMES_WIDTH = 16

INPUT_DROPOUT = 0.0
POST_RNN_DROPOUT = 0.0
NOISE_LEVEL = 0.0

AUTO_POST_WIDTH = [ 128 ]

DENSE_L2 = 0.0

GAUSSIAN_POOLING_VAR = 1.0
GAUSSIAN_POOLING_LEN_DELTA = 3.0

RNN_WIDTH = 32
DENSE_POST_WIDTH = [ (128, 0.0) ]
FEATURE_COUNT = 32

ANNEAL_MAX_STEP = 10000.0

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
    self.use_sphereface = False

    self.margin = 0.35
    self.radius = 9.2

    self.embedding = Embedding('embedding', dataset.MAX_CHAR + 2, EMBED_WIDTH)

    self.rnn_cell = tf.contrib.rnn.LSTMBlockCell(name='lstm_cell',
        num_units=RNN_WIDTH)

    self.rnn_rev_cell = tf.contrib.rnn.LSTMBlockCell(name='lstm_rev_cell',
        num_units=RNN_WIDTH)

    # Just to convert rnn_rev output into holds+deltas
    self.post_rev = tf.layers.Dense(name='dense_post_rev',
                                    units=2,
                                    activation=tf.nn.relu,
                                    kernel_regularizer=self.l2)

    self.input_dropout = tf.keras.layers.Dropout(name='input_dropout',
        rate=INPUT_DROPOUT)
    self.post_rnn_dropout = tf.keras.layers.Dropout(name='post_rnn_dropout',
        rate=POST_RNN_DROPOUT)

    self.process_times = tf.layers.Dense(name='process_times',
                                         units=TIMES_WIDTH,
                                         activation=tf.nn.relu,
                                         kernel_regularizer=self.l2)

    self.auto_post = []
    for i, width in enumerate(AUTO_POST_WIDTH):
      dense = tf.layers.Dense(name='auto_post_{}'.format(i),
                              units=width,
                              activation=tf.nn.relu,
                              kernel_regularizer=self.l2)
      self.auto_post.append({ 'dense': dense })

    self.post = []
    for i, (width, dropout) in enumerate(DENSE_POST_WIDTH):
      dense = tf.layers.Dense(name='dense_post_{}'.format(i),
                              units=width,
                              activation=tf.nn.relu,
                              kernel_regularizer=self.l2)
      dropout = tf.keras.layers.Dropout(name='dropout_post_{}'.format(i),
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
    noise = tf.random.normal(tf.shape(times))
    times += tf.where(self.training, NOISE_LEVEL, 0.0) * noise

    # Process holds+deltas
    times = self.process_times(times)

    series = tf.concat([ times, embedding ], axis=-1, name='full_input')
    series = self.input_dropout(series, training=self.training)

    return series, embedding

  def build(self, holds, codes, deltas, sequence_len=None, auto=False):
    batch_size = tf.shape(codes)[0]
    max_sequence_len = int(codes.shape[1])
    if sequence_len is None:
      sequence_len = tf.constant(max_sequence_len, dtype=tf.int32,
          shape=(1,))
      sequence_len = tf.tile(sequence_len, [ batch_size ])

    series, embedding = self.apply_embedding(holds, codes, deltas)
    series = tf.unstack(series, axis=1, name='unstacked_series')

    outputs, state = tf.nn.static_rnn(
          cell=self.rnn_cell,
          dtype=tf.float32,
          inputs=series)

    if auto:
      state = tf.nn.rnn_cell.LSTMStateTuple(
          c=self.post_rnn_dropout(state.c, training=self.training),
          h=self.post_rnn_dropout(state.h, training=self.training))
      embedding = tf.unstack(embedding, axis=1, name='unstacked_embedding')
      outputs, _ = tf.nn.static_rnn(
            cell=self.rnn_rev_cell,
            dtype=tf.float32,
            inputs=embedding,
            initial_state=state)
      x = tf.stack(outputs, axis=1, name='stacked_rev_outputs')
      for entry in self.auto_post:
        x = entry['dense'](x)
      x = self.post_rev(x)
      return x

    outputs = tf.stack(outputs, axis=1, name='stacked_outputs')
    x = tf.reduce_mean(outputs, axis=1, name='avg_output')
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
    for percentile in [ 5, 10, 25, 50, 75, 90, 95 ]:
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

    return positive_distances, negative_distances, metrics


  # As in https://arxiv.org/pdf/1703.07464.pdf
  # More like in: http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_CosFace_Large_Margin_CVPR_2018_paper.pdf
  # TODO(indutny): try http://ydwen.github.io/papers/WenECCV16.pdf
  # TODO(indutny): try http://openaccess.thecvf.com/content_cvpr_2018/papers/Zheng_Ring_Loss_Convex_CVPR_2018_paper.pdf
  # TODO(indutny): try https://arxiv.org/pdf/1704.08063.pdf
  # TODO(indutny): try https://arxiv.org/pdf/1703.09507.pdf
  # See http://proceedings.mlr.press/v48/liud16.pdf
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

        # TODO(indutny): try annealing again
        positive_distances = psi
      else:
        positive_distances -= self.margin

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

  def get_auto_loss(self, holds, deltas, outputs):
    with tf.name_scope('auto_loss', [ holds, deltas, outputs ]):
      pred_holds, pred_deltas = tf.split(outputs, [ 1, 1 ], axis=-1)
      pred_holds = tf.squeeze(pred_holds, axis=-1)
      pred_deltas = tf.squeeze(pred_deltas, axis=-1)

      # Mean hold over sequence
      hold_mean = tf.reduce_mean(holds, axis=-1, keepdims=True) + 1e-23
      delta_mean = tf.reduce_mean(deltas, axis=-1, keepdims=True) + 1e-23

      # Mean Square Loss
      hold_loss = tf.reduce_mean(
          ((pred_holds - holds) / hold_mean) ** 2.0, axis=-1)
      delta_loss = tf.reduce_mean(
          ((pred_deltas - deltas) / delta_mean) ** 2.0, axis=-1)

      hold_loss = tf.reduce_mean(hold_loss)
      delta_loss = tf.reduce_mean(delta_loss)

      loss = 1.0 / 2.0 * (hold_loss + delta_loss)

      metrics = {}
      metrics['loss'] = loss
      metrics['hold_loss'] = hold_loss
      metrics['delta_loss'] = delta_loss
      return metrics
