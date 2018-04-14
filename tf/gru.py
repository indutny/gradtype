import tensorflow as tf

def hard_sigmoid(x):
  x = (0.2 * x) + 0.5
  zero = tf.convert_to_tensor(0., dtype=tf.float32)
  one = tf.convert_to_tensor(1., dtype=tf.float32)
  x = tf.clip_by_value(x, zero, one)
  return x

# Simplified port of GRU from Keras
class GRUCell():
  def __init__(self, units, name, training, recurrent_dropout=0.3):
    self.units = units
    self.name = name
    self.kernel_initializer = tf.glorot_uniform_initializer()
    self.recurrent_initializer = tf.initializers.orthogonal()
    self.bias_initializer = tf.initializers.zeros()
    self.activation = tf.nn.tanh
    self.recurrent_activation = hard_sigmoid
    self.l1 = tf.contrib.layers.l1_regularizer(0.005)
    self.l2 = tf.contrib.layers.l2_regularizer(0.01)
    self.regularizer = tf.contrib.layers.sum_regularizer([ self.l1, self.l2 ])
    self.recurrent_keep = 1.0 - \
        tf.cast(training, tf.float32) * recurrent_dropout

  def build(self, input_shape):
    with tf.variable_scope(None, default_name=self.name):
      kernel_shape = (input_shape[1], self.units,)
      bias_shape = (self.units,)

      self.kernel_z = tf.get_variable('kernel_z', shape=kernel_shape,
          initializer=self.kernel_initializer,
          regularizer=self.regularizer)
      self.kernel_r = tf.get_variable('kernel_r', shape=kernel_shape,
          initializer=self.kernel_initializer,
          regularizer=self.regularizer)
      self.kernel_h = tf.get_variable('kernel_h', shape=kernel_shape,
          initializer=self.kernel_initializer,
          regularizer=self.regularizer)
      self.bias_z = tf.get_variable('bias_z', shape=bias_shape,
          initializer=self.bias_initializer)
      self.bias_r = tf.get_variable('bias_r', shape=bias_shape,
          initializer=self.bias_initializer)
      self.bias_h = tf.get_variable('bias_h', shape=bias_shape,
          initializer=self.bias_initializer)

      recurrent_shape = (self.units, self.units,)

      self.recurrent_z = tf.get_variable('recurrent_z', shape=recurrent_shape,
          initializer=self.recurrent_initializer,
          regularizer=self.regularizer)
      self.recurrent_r = tf.get_variable('recurrent_r', shape=recurrent_shape,
          initializer=self.recurrent_initializer,
          regularizer=self.regularizer)
      self.recurrent_h = tf.get_variable('recurrent_h', shape=recurrent_shape,
          initializer=self.recurrent_initializer,
          regularizer=self.regularizer)

      return tf.expand_dims(tf.zeros(shape=(self.units,)), axis=0)

  def __call__(self, inputs, state, training=False):
    with tf.name_scope(self.name):
      # TODO(indutny): port dropout
      inputs_z = inputs
      inputs_r = inputs
      inputs_h = inputs

      x_z = tf.matmul(inputs_z, self.kernel_z) + self.bias_z
      x_r = tf.matmul(inputs_r, self.kernel_r) + self.bias_r
      x_h = tf.matmul(inputs_h, self.kernel_h) + self.bias_h

      h_tm1 = state
      h_tm1_z = tf.nn.dropout(h_tm1, self.recurrent_keep)
      h_tm1_r = tf.nn.dropout(h_tm1, self.recurrent_keep)
      h_tm1_h = tf.nn.dropout(h_tm1, self.recurrent_keep)

      recurrent_z = tf.matmul(h_tm1_z, self.recurrent_z)
      recurrent_r = tf.matmul(h_tm1_r, self.recurrent_r)

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      recurrent_h = tf.matmul(r * h_tm1_h, self.recurrent_h)
      hh = self.activation(x_h + recurrent_h)

      h = z * h_tm1 + (1 - z) * hh
      return h, h
