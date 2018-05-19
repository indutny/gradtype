import math
import tensorflow as tf

# Internal
import dataset

EMBED_WIDTH = 11
DELTA_WIDTH = 5

INPUT_DROPOUT = 0.0
RNN_INPUT_DROPOUT = 0.05
RNN_STATE_DROPOUT = 0.5
RNN_OUTPUT_DROPOUT = 0.0
RNN_USE_RESIDUAL = False
RNN_USE_BIDIR = False

DENSE_L2 = 0.001
CNN_L2 = 0.0

RNN_WIDTH = [ 32 ]
DENSE_POST_WIDTH = [ ]
FEATURE_COUNT = 7

CNN_WIDTH = [ 8, 8, 8, 16, 16, 16, 32, 32 ]

class Model():
  def __init__(self, training):
    self.l2 = tf.contrib.layers.l2_regularizer(DENSE_L2)
    self.cnn_l2 = tf.contrib.layers.l2_regularizer(CNN_L2)
    self.training = training
    self.use_pooling = False
    self.random_len = False # for now

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

  def build(self, rows):
    if RNN_USE_BIDIR:
      outputs, _, _ = tf.nn.dynamic_bidirectional_rnn(
          cell_fw=self.rnn_cell_fw,
          cell_bw=self.rnn_cell_bw,
          dtype=tf.float32,
          inputs=rows)
    else:
      outputs, _ = tf.nn.dynamic_rnn(
          cell=self.rnn_cell_fw,
          dtype=tf.float32,
          inputs=rows)

    x = outputs[:, -1]

    for post in self.post:
      x = post(x)

    x = self.features(x)
    # x = tf.nn.l2_normalize(x, axis=-1)

    return x

  def build_conv(self, series):
    sequence_len = int(series.shape[1])

    def causal_padding(series):
      current_sequence_len = int(series.shape[1])
      if sequence_len == current_sequence_len:
        return series
      to_pad = sequence_len - current_sequence_len
      print('padded', to_pad)

      return tf.pad(series, [ [ 0, 0 ], [ to_pad, 0 ], [ 0, 0 ] ])

    def residual_block(i, width, dilation, series):
      with tf.name_scope('residual_block_{}'.format(i), [ series ]):
        x = series

        x = tf.layers.conv1d(x, filters=width, kernel_size=5,
                             dilation_rate=dilation, activation=tf.nn.selu,
                             kernel_regularizer=self.cnn_l2)
        x = causal_padding(x)
        x = tf.layers.dropout(x, rate=0.2, training=self.training)

        x = tf.layers.conv1d(x, filters=width, kernel_size=5,
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
    # x = tf.nn.l2_normalize(x, axis=-1)
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
      category_mask):
    positives = tf.gather(proxies, categories, axis=0,
        name='positive_proxies')

    negative_masks = tf.one_hot(categories, category_count, on_value=False,
        off_value=True, name='negative_mask')
    negative_masks = tf.logical_and(negative_masks, \
        tf.expand_dims(category_mask, axis=0))

    def apply_mask(mask):
      return tf.boolean_mask(proxies, mask, axis=0, name='batch_negatives')

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

    return positive_distances, negative_distances, metrics


  # As in https://arxiv.org/pdf/1703.07464.pdf
  def get_proxy_loss(self, output, categories, weights, category_count, \
      category_mask):
    with tf.name_scope('proxy_loss', [ output, categories, weights, \
        category_mask ]):
      proxies = tf.get_variable('points',
          shape=(category_count, FEATURE_COUNT,))
      proxies = tf.nn.l2_normalize(proxies, axis=-1)

      weights = tf.gather(weights, categories, axis=0, \
          name='per_category_weight')

      positive_distances, negative_distances, metrics = self.get_proxy_common( \
          proxies, output, categories, category_count, category_mask)

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
        return tf.nn.l2_normalize(tf.reduce_mean(points, axis=0), axis=-1)

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
