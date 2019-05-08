import math
import tensorflow as tf

# Internal
import dataset

EMBED_WIDTH = 11
DELTA_WIDTH = 5

INPUT_DROPOUT = 0.0
RNN_INPUT_DROPOUT = 0.0
RNN_STATE_DROPOUT = 0.0
RNN_OUTPUT_DROPOUT = 0.0
RNN_USE_RESIDUAL = False
RNN_USE_BIDIR = False

DENSE_L2 = 0.0
CNN_L2 = 0.0

RNN_WIDTH = [ 64 ]
DENSE_POST_WIDTH = [ 64 ]
FEATURE_COUNT = 64

CNN_WIDTH = [ 64, 64, 64 ]

TIME_LAMBDA = 0.0

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
    self.use_pooling = False
    self.random_len = False

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

    self.post = []
    for i, width in enumerate(DENSE_POST_WIDTH):
      self.post.append(tf.layers.Dense(name='dense_post_{}'.format(i),
                                       units=width,
                                       activation=tf.nn.selu,
                                       kernel_regularizer=self.l2))

    self.features = tf.layers.Dense(name='features',
                                    units=FEATURE_COUNT,
                                    kernel_regularizer=self.l2)

  def apply_embedding(self, holds, codes, deltas, return_raw=False):
    embedding = self.embedding.apply(codes)
    holds = tf.expand_dims(holds, axis=-1, name='expanded_holds')
    deltas = tf.expand_dims(deltas, axis=-1, name='expanded_deltas')

    times = tf.concat([ holds, deltas ], axis=-1, name='times')

    # Add random noise for training
    # TODO(indutny): this should be exponential, not poisson
    noise = tf.random.poisson(TIME_LAMBDA, tf.shape(times), name='times_noise')
    times = tf.where(self.training, times + noise, times)

    # Process holds+deltas
    times = tf.layers.conv1d(times, filters=DELTA_WIDTH, kernel_size=1,
                             activation=tf.nn.selu,
                             kernel_regularizer=self.l2,
                             name='processed_times')

    series = tf.concat([ times, embedding ], axis=-1, name='full_input')
    series = tf.layers.dropout(series, rate=INPUT_DROPOUT,
        training=self.training)

    if return_raw:
      return series, embedding

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

    if self.use_pooling:
      x = tf.reduce_mean(stacked_output, axis=1, name='output')
    elif self.random_len:
      max_index = sequence_len - 1
      random_len = tf.random_uniform(shape=(batch_size,),
          minval=int(max_sequence_len / 2),
          maxval=max_sequence_len,
          dtype=tf.int32,
          name='random_len')
      selected_len = tf.where(self.training, random_len, max_index,
          name='selected_len')

      def select_random(pair):
        outputs = pair[0]
        index = pair[1]
        max_index = pair[2]

        index = tf.minimum(index, max_index, name='clamped_index')

        return tf.gather(outputs, index, axis=0, name='select_random')

      x = tf.map_fn(select_random,
          (stacked_output, selected_len, max_index, ),
          dtype=tf.float32)
    else:
      def select_last(pair):
        outputs = pair[0]
        seq_len = pair[1]
        return tf.gather(outputs, seq_len, axis=0, name='select_last')

      x = tf.map_fn(select_last, (stacked_output, sequence_len - 1,),
          dtype=tf.float32)

    for post in self.post:
      x = post(x)

    x = self.features(x)

    return x

  def build_conv(self, holds, codes, deltas):
    series = self.apply_embedding(holds, codes, deltas)
    sequence_len = int(deltas.shape[1])

    def causal_padding(series):
      current_sequence_len = int(series.shape[1])
      if sequence_len == current_sequence_len:
        return series
      to_pad = sequence_len - current_sequence_len

      return tf.pad(series, [ [ 0, 0 ], [ to_pad, 0 ], [ 0, 0 ] ])

    def residual_block(i, width, dilation, series):
      with tf.name_scope('residual_block_{}'.format(i), [ series ]):
        x = series

        x = tf.layers.conv1d(x, filters=width, kernel_size=3,
                             dilation_rate=dilation, activation=tf.nn.selu,
                             kernel_regularizer=self.cnn_l2)
        x = causal_padding(x)
        x = tf.layers.dropout(x, rate=0.2, training=self.training)

        x = tf.layers.conv1d(x, filters=width, kernel_size=3,
                             dilation_rate=dilation, activation=tf.nn.selu,
                             kernel_regularizer=self.cnn_l2)
        x = causal_padding(x)
        x = tf.layers.dropout(x, rate=0.2, training=self.training)

        if series.shape[2] != x.shape[2]:
          series = tf.layers.conv1d(x, filters=x.shape[2], kernel_size=1,
                                    kernel_regularizer=self.cnn_l2)

        return tf.nn.selu(series + x)

    for i, width in enumerate(CNN_WIDTH):
      series = residual_block(i, width, 2 ** i, series)

    x = series[:, -1]

    x = self.features(x)
    return x

  def build_auto(self, codes, deltas):
    series, embeddings = self.apply_embedding(codes, deltas, return_raw=True)

    frames = tf.unstack(series, axis=1, name='unstacked_output')

    outputs, _ = tf.nn.static_rnn(cell=self.rnn_cell_fw, dtype=tf.float32,
        inputs=frames)

    decoder = tf.contrib.rnn.LSTMBlockCell(name='decoder',
        num_units=RNN_WIDTH[-1])
    decoder = tf.contrib.rnn.DropoutWrapper(decoder,
        state_keep_prob=tf.where(self.training, 1.0 - 0.0, 1.0))

    batch_size = tf.shape(codes)[0]
    decoder_state = decoder.zero_state(batch_size, dtype=tf.float32)
    decoder_state = tf.contrib.rnn.LSTMStateTuple(outputs[-1], decoder_state.h)

    unstacked_embeddings = tf.unstack(embeddings, axis=1, \
        name='unstacked_embeddings')
    outputs, _ = tf.nn.static_rnn(cell=decoder, inputs=unstacked_embeddings,
        initial_state=decoder_state)

    mux = tf.layers.Dense(name='mux', units=1, kernel_regularizer=self.l2)
    outputs = [ mux(output) for output in outputs ]

    outputs = tf.stack(outputs, axis=1, name='stacked_decoder_output')
    return outputs, deltas

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
      accuracy /= tf.reduce_sum(batch_weights)
      accuracy = tf.reduce_sum(accuracy)

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

  # TODO(indutny): use `weights`?
  def get_proxy_common(self, proxies, output, categories, category_count, \
      category_mask, adversarial_count=None):
    if adversarial_count != None:
      adversarial = output[-adversarial_count:]
      output = output[:-adversarial_count]

    positives = tf.gather(proxies, categories, axis=0,
        name='positive_proxies')

    negative_masks = tf.one_hot(categories, category_count, on_value=False,
        off_value=True, name='negative_mask')
    negative_masks = tf.logical_and(negative_masks, \
        tf.expand_dims(category_mask, axis=0))

    def apply_mask(mask):
      negatives = tf.boolean_mask(proxies, mask, axis=0, \
          name='batch_negatives')

      # TODO(indutny): optimize?
      if adversarial_count != None:
        negatives = tf.concat([ negatives, adversarial ], axis=0,
            name='batch_negatives_and_adversaries')
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
    metrics['ratio_25',] = metrics['negative_25'] / \
        (metrics['positive_75'] + epsilon)
    metrics['ratio_10',] = metrics['negative_10'] / \
        (metrics['positive_90'] + epsilon)
    metrics['ratio_5',] = metrics['negative_5'] / \
        (metrics['positive_95'] + epsilon)

    return positive_distances, negative_distances, metrics


  # As in https://arxiv.org/pdf/1703.07464.pdf
  def get_proxy_loss(self, output, categories, weights, category_count, \
      category_mask, adversarial_count=None):
    with tf.name_scope('proxy_loss', [ output, categories, weights, \
        category_mask ]):
      proxies = tf.get_variable('points',
          initializer=tf.initializers.orthogonal(),
          shape=(category_count, FEATURE_COUNT,))

      weights = tf.gather(weights, categories, axis=0, \
          name='per_category_weight')

      positive_distances, negative_distances, metrics = self.get_proxy_common( \
          proxies, output, categories, category_count, category_mask,
          adversarial_count)

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

  def get_proxy_val_metrics(self, output, categories, weights, category_count, \
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

  def get_auto_metrics(self, output, deltas, categories, weights):
    with tf.name_scope('auto_metrics', [ output, deltas, categories, \
        weights ]):
      batch_weights = tf.gather(weights, categories, axis=0, \
          name='per_category_weight')

      output = tf.squeeze(output, axis=2)
      loss = batch_weights * tf.reduce_mean((deltas - output) ** 2, axis=-1)
      loss = tf.reduce_mean(loss, axis=0, name='mean_loss')

      metrics = {}
      metrics['loss'] = loss

      # Add batch dimension
      embedding = tf.expand_dims(self.embedding.weights, axis=0)
      # Add color dimension
      embedding = tf.expand_dims(embedding, axis=-1)
      embedding = tf.summary.image('embedding', embedding)

      return metrics, tf.summary.merge([ embedding ])
