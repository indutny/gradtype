import tensorflow as tf

# Internal
import dataset

EMBED_WIDTH = 127
DENSE_PRE_COUNT = 0
DENSE_PRE_WIDTH = 32
DENSE_PRE_RESIDUAL_COUNT = 0

INPUT_DROPOUT = 0.0
RNN_STATE_DROPOUT = 0.2
RNN_OUTPUT_DROPOUT = 0.0

DENSE_L2 = 0.001
CNN_L2 = 0.004

# See: https://arxiv.org/pdf/1708.01009.pdf
RNN_ACTIVATION_L2 = 0.0
RNN_TEMP_ACTIVATION_L2 = 0.0

RNN_WIDTH = [ 128, 128, 128 ]
DENSE_POST_WIDTH = [ ]
FEATURE_COUNT = 128

class Embedding():
  def __init__(self, name, max_code, width, regularizer=None):
    self.name = name
    self.width = width
    with tf.variable_scope(None, default_name=self.name):
      self.weights = tf.get_variable('weights', shape=(max_code, width),
                                     regularizer=regularizer)

  def apply(self, codes):
    with tf.name_scope(None, values=[ codes ], default_name=self.name):
      return tf.gather(self.weights, codes)

class Model():
  def __init__(self, training):
    self.l2 = tf.contrib.layers.l2_regularizer(DENSE_L2)
    self.cnn_l2 = tf.contrib.layers.l2_regularizer(CNN_L2)
    self.training = training
    self.use_pooling = True

    self.embedding = Embedding('embedding', dataset.MAX_CHAR + 2, EMBED_WIDTH,
                               regularizer=self.l2)

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

    cells = []
    states = []
    for i, width in enumerate(RNN_WIDTH):
      cell = tf.contrib.rnn.GRUBlockCellV2(name='gru_{}'.format(i), \
          num_units=width)

      states.append(tf.get_variable('initial_state_{}'.format(i), \
          shape=(cell.state_size, ),
          regularizer=self.l2))

      cell = tf.contrib.rnn.DropoutWrapper(cell,
          state_keep_prob=tf.where(training, 1.0 - RNN_STATE_DROPOUT, 1.0),
          output_keep_prob=tf.where(training, 1.0 - RNN_OUTPUT_DROPOUT, 1.0))

      cell = tf.contrib.rnn.ResidualWrapper(cell)

      cells.append(cell)
    self.rnn_cells = cells
    self.rnn_states = states

    self.post = []
    for i, width in enumerate(DENSE_POST_WIDTH):
      self.post.append(tf.layers.Dense(name='dense_post_{}'.format(i),
                                       units=width,
                                       activation=tf.nn.selu,
                                       kernel_regularizer=self.l2))

    self.features = tf.layers.Dense(name='features',
                                    units=FEATURE_COUNT,
                                    kernel_regularizer=self.l2)

  def apply_embedding(self, codes, deltas):
    embedding = self.embedding.apply(codes)
    deltas = tf.expand_dims(deltas, axis=-1, name='expanded_deltas')

    series = tf.concat([ deltas, embedding ], axis=-1, name='full_input')
    series = tf.layers.dropout(series, rate=INPUT_DROPOUT,
        training=self.training)

    return series

  def build(self, codes, deltas):
    batch_size = tf.shape(codes)[0]
    sequence_len = int(codes.shape[1])

    series = self.apply_embedding(codes, deltas)
    frames = tf.unstack(series, axis=1, name='unstacked_output')

    new_frames = []
    for frame in frames:
      for pre in self.pre:
        frame = pre(frame)

      for [ minor, major ] in self.pre_residual:
        frame = tf.nn.selu(frame + major(minor(frame)))

      new_frames.append(frame)
    frames = new_frames

    # Add [ batch_size, ] dimension
    states = [
        tf.tile(tf.expand_dims(state, axis=0), (batch_size, 1, ))
        for state in self.rnn_states
    ]

    states = [
        state + tf.cast(self.training, tf.float32) * \
            tf.random_normal(tf.shape(state), stddev=0.01)
        for state in states
    ]

    for i, cell, state in zip(range(len(states)), self.rnn_cells, states):
      outputs, _ = tf.nn.static_rnn( \
          cell=cell,
          initial_state=state,
          inputs=frames)
      stacked_output = tf.stack(outputs, axis=1, name='stacked_output')

      if RNN_ACTIVATION_L2 != 0.0:
        l2 = stacked_output ** 2
        # Sum over activations
        l2 = tf.reduce_sum(l2, axis=-1)
        # Mean over time dimension
        l2 = tf.reduce_mean(l2, axis=1)
        # Mean over batch dimensions
        l2 = tf.reduce_mean(l2, axis=0)
        l2 *= RNN_ACTIVATION_L2

        tf.losses.add_loss(l2, tf.GraphKeys.REGULARIZATION_LOSSES)

      if RNN_TEMP_ACTIVATION_L2 != 0.0:
        left = stacked_output[:, :-1]
        right = stacked_output[:, 1:]
        l2 = (left - right) ** 2
        # Sum over activations
        l2 = tf.reduce_sum(l2, axis=-1)
        # Mean over time dimension
        l2 = tf.reduce_mean(l2, axis=1)
        # Mean over batch dimensions
        l2 = tf.reduce_mean(l2, axis=0)
        l2 *= RNN_TEMP_ACTIVATION_L2
        tf.losses.add_loss(l2, tf.GraphKeys.REGULARIZATION_LOSSES)

      frames = outputs

    if self.use_pooling:
      x = tf.reduce_max(stacked_output, axis=1, name='output')
    else:
      x = outputs[-1]

    for post in self.post:
      x = post(x)

    x = self.features(x)
    x = tf.nn.l2_normalize(x, axis=-1)

    return x

  def build_conv(self, codes, deltas):
    series = self.apply_embedding(codes, deltas)

    series = tf.layers.conv1d(series, filters=96, kernel_size=16,
                              activation=tf.nn.selu,
                              kernel_regularizer=self.cnn_l2)
    series = tf.layers.conv1d(series, filters=96, kernel_size=10,
                              activation=tf.nn.selu,
                              kernel_regularizer=self.cnn_l2)
    series = tf.layers.conv1d(series, filters=96, kernel_size=8,
                              activation=tf.nn.selu,
                              kernel_regularizer=self.cnn_l2)

    x = tf.squeeze(series, axis=1)
    x = tf.layers.dropout(x, training=self.training)

    x = self.features(x)
    x = tf.nn.l2_normalize(x, axis=-1)
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

      all_positives = tf.boolean_mask(distances, same_mask)
      all_negatives = tf.boolean_mask(distances, not_same_mask)

      metrics = {}
      metrics['loss'] = loss
      metrics['positive_25'] = tf.contrib.distributions.percentile( \
          all_positives,
          25.0,
          name='positive_25')
      metrics['positive_50'] = tf.reduce_mean(all_positives, \
          name='positive_50')
      metrics['positive_75'] = tf.contrib.distributions.percentile( \
          all_positives,
          75.0,
          name='positive_75')
      metrics['positive_90'] = tf.contrib.distributions.percentile( \
          all_positives,
          90.0,
          name='positive_90')

      metrics['negative_50'] = tf.reduce_mean(all_negatives, \
          name='mean_negative')
      metrics['negative_25'] = tf.contrib.distributions.percentile( \
          all_negatives,
          25.0,
          name='negative_25')
      metrics['negative_75'] = tf.contrib.distributions.percentile( \
          all_negatives,
          75.0,
          name='negative_75')
      metrics['negative_90'] = tf.contrib.distributions.percentile( \
          all_negatives,
          90.0,
          name='negative_90')

      metrics['active_triplets'] = \
          tf.reduce_mean(tf.cast(tf.greater(triplet_distance, zero), \
                                 tf.float32),
                         name='active_triplets')
      metrics['norm'] = tf.sqrt(tf.reduce_mean(distances2, name='norm'))

      return metrics

  def get_regression_metrics(self, output, categories, weights):
    with tf.name_scope('regression_loss', [ output, categories, weights ]):
      categories_one_hot = tf.one_hot(categories, output.shape[1], axis=-1)

      batch_weights = tf.gather(weights, categories, axis=0, \
          name='per_category_weight')

      loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, \
          labels=categories_one_hot)
      loss *= batch_weights
      loss = tf.reduce_mean(loss)

      predictions = tf.cast(tf.argmax(output, axis=-1), tf.int32)

      accuracy = tf.equal(predictions, tf.cast(categories, tf.int32))
      accuracy = tf.cast(accuracy, tf.float32) * batch_weights
      accuracy = tf.reduce_mean(accuracy)

      confusion = tf.confusion_matrix(categories, predictions,
                                      num_classes=tf.shape(weights)[0],
                                      dtype=tf.float32)
      confusion *= weights

      # Add batch dimension
      confusion = tf.expand_dims(confusion, axis=0)
      # Add color dimension
      confusion = tf.expand_dims(confusion, axis=-1)

      confusion = tf.summary.image('confusion', confusion)

      metrics = {}
      metrics['loss'] = loss
      metrics['accuracy'] = accuracy

      return metrics, tf.summary.merge([ confusion ])

  def get_proxy_common(self, proxies, output, categories, category_count):
    positives = tf.gather(proxies, categories, axis=0,
        name='positive_proxies')

    negative_masks = tf.one_hot(categories, category_count, on_value=False,
        off_value=True, name='negative_mask')

    def apply_mask(mask):
      return tf.boolean_mask(proxies, mask, axis=0, name='batch_negatives')

    negatives = tf.map_fn(apply_mask, negative_masks, name='negatives',
        dtype=tf.float32)

    positive_distances = positives - output
    negative_distances = negatives - tf.expand_dims(output, axis=1)

    positive_distances **= 2
    negative_distances **= 2

    positive_distances = tf.reduce_sum(positive_distances, axis=-1,
        name='positive_distances')
    negative_distances = tf.reduce_sum(negative_distances, axis=-1,
        name='negative_distances')

    metrics = {}
    for percentile in [ 5, 25, 50, 75, 95 ]:
      neg_p = tf.contrib.distributions.percentile(negative_distances,
          percentile, name='negative_{}'.format(percentile))
      metrics['negative_{}'.format(percentile)] = neg_p

      pos_p = tf.contrib.distributions.percentile(positive_distances,
          percentile, name='positive_{}'.format(percentile))
      metrics['positive_{}'.format(percentile)] = pos_p

    return positive_distances, negative_distances, metrics


  # As in https://arxiv.org/pdf/1703.07464.pdf
  def get_proxy_loss(self, output, categories, weights, category_count):
    with tf.name_scope('proxy_loss', [ output, categories, weights ]):
      proxies = tf.get_variable('points',
          shape=(category_count, FEATURE_COUNT,))
      proxies = tf.nn.l2_normalize(proxies, axis=-1)

      weights = tf.gather(weights, categories, axis=0, \
          name='per_category_weight')

      positive_distances, negative_distances, metrics = self.get_proxy_common( \
          proxies, output, categories, category_count)

      exp_pos = tf.exp(-positive_distances, name='exp_pos')
      exp_neg = tf.exp(-negative_distances, name='exp_neg')

      total_exp_neg = tf.reduce_sum(exp_neg, axis=-1, name='total_exp_neg')

      epsilon = 1e-12
      ratio = exp_pos / (total_exp_neg + epsilon)

      loss = -tf.log(ratio + epsilon, name='loss_vector')
      loss *= weights
      loss = tf.reduce_mean(loss, name='loss')

      metrics['loss'] = loss

      return metrics

  def get_proxy_val_metrics(self, output, categories, weights, category_count):
    with tf.name_scope('proxy_val_metrics', [ output, categories ]):
      # Compute proxies as mean points
      def compute_mean_proxy(category):
        points = tf.boolean_mask(output, tf.equal(categories, category),
            'category_points')
        return tf.reduce_mean(points, axis=0)

      proxies = tf.map_fn(compute_mean_proxy, tf.range(category_count),
          dtype=tf.float32)

      _, _, metrics = self.get_proxy_common(proxies, output, categories, \
          category_count)

      return metrics
