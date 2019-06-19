import math
import tensorflow as tf

# Internal
import dataset

LITTLE_EMBED_WIDTH = 4
GRID_WIDTH = 28

DENSE_L2 = 0.0

FEATURE_COUNT = 32

ANNEAL_MAX_STEP = 100.0
MAX_SPHERE_STRENGTH = 0.9

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

    self.little_embedding = tf.keras.layers.Embedding(
        name='little_embedding',
        input_dim=dataset.MAX_CHAR + 2,
        output_dim=LITTLE_EMBED_WIDTH)

    self.grid_embedding = tf.keras.layers.Embedding(
        name='grid_embedding',
        input_dim=dataset.MAX_CHAR + 2,
        output_dim=GRID_WIDTH)

    self.phase_freq = tf.layers.Dense(
        name='phase',
        units=2 * GRID_WIDTH,
        activation=tf.nn.relu,
        kernel_regularizer=self.l2)

    self.conv = [
        tf.keras.layers.Conv2D(name='conv_1', filters=8, kernel_size=3,
            activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(name='pool_1', pool_size=(2,2)),
        tf.keras.layers.Conv2D(name='conv_2', filters=16, kernel_size=3,
            activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(name='pool_2', pool_size=(2,2)),
        tf.keras.layers.Conv2D(name='conv_3', filters=32, kernel_size=3,
            activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(name='pool_3', pool_size=(2,2)),

        tf.keras.layers.Conv2D(name='features', filters=FEATURE_COUNT,
            kernel_size=1),
    ]

  def build(self, holds, codes, deltas, sequence_len=None):
    batch_size = tf.shape(codes)[0]
    max_sequence_len = int(codes.shape[1])
    if sequence_len is None:
      sequence_len = tf.constant(max_sequence_len, dtype=tf.int32,
          shape=(1,))
      sequence_len = tf.tile(sequence_len, [ batch_size ])

    sequence_len = tf.cast(sequence_len, dtype=tf.float32)

    little_embedding = self.little_embedding(codes)

    index = tf.expand_dims(
        tf.range(max_sequence_len, dtype=tf.float32),
        axis=0,
        name='index')
    mask = tf.cast(index < sequence_len, dtype=tf.float32, name='mask')

    cont_index = tf.expand_dims(index / tf.expand_dims(sequence_len, axis=-1),
        axis=-1,
        name='cont_index')
    cont_index *= 2.0 * math.pi

    times = tf.concat([
        tf.expand_dims(holds, axis=-1), tf.expand_dims(deltas, axis=-1) ],
        axis=-1,
        name='times')
    phase_input = tf.concat(
        [ tf.sin(cont_index), tf.cos(cont_index), times, little_embedding ],
        axis=-1,
        name='phase_input')

    phase_freq = self.phase_freq(phase_input)

    phase, freq = tf.split(phase_freq, [ GRID_WIDTH, GRID_WIDTH ], axis=-1)

    grid = tf.range(GRID_WIDTH, dtype=tf.float32) / float(GRID_WIDTH)
    grid = tf.reshape(grid, shape=[ 1, 1, GRID_WIDTH ], name='grid')

    grid *= freq
    grid += phase
    grid = tf.expand_dims(grid, axis=-1)

    grid = tf.concat([
        tf.sin(grid, name='sin_grid'),
        tf.cos(grid, name='cos_grid'),
    ], axis=-1, name='full_grid')

    grid = tf.expand_dims(grid, axis=2)

    embedding = self.grid_embedding(codes)

    grid *= tf.expand_dims(tf.expand_dims(embedding, axis=-1), axis=-1)
    # TODO(indutny): apply mask
    grid = tf.reduce_mean(grid, axis=1, name='sum_grid')

    x = grid
    for l in self.conv:
      if isinstance(l, tf.keras.layers.Activation):
        x = l(x, training=self.training)
      else:
        x = l(x)
      print(x)

    x = tf.reshape(x, shape=(batch_size, FEATURE_COUNT,))
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
