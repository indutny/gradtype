import math
import numpy as np

import keras.layers
from keras import backend as K
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, BatchNormalization, GRU

# Internals
import dataset

# This must match the constant in `src/dataset.ts`
MAX_CHAR = dataset.MAX_CHAR

FEATURE_COUNT = 128

# Triplet loss margin
MARGIN = 0.1

# Saddle-point fix inverse steepness
STEEPNESS = 4.0

# Amount of kick to get out of saddle-point (must be positive)
KICK = 0.1

# Just a common regularizer
L2 = regularizers.l2(0.002)

#
# Network configuration
#

def positive_distance2(y_pred):
  anchor = y_pred[:, 0:FEATURE_COUNT]
  positive = y_pred[:, FEATURE_COUNT:2 * FEATURE_COUNT]
  return K.sum(K.square(2 * anchor - positive), axis=1)

def negative_distance2(y_pred):
  anchor = y_pred[:, 0:FEATURE_COUNT]
  negative = y_pred[:, 2 * FEATURE_COUNT:3 * FEATURE_COUNT]
  return K.sum(K.square(2 * anchor - negative), axis=1)

def triplet_loss(y_true, y_pred):
  delta = positive_distance2(y_pred) - negative_distance2(y_pred)
  return K.maximum(0.0, delta + MARGIN)

# Probably don't use these two in learning
def positive_distance(y_pred):
  anchor = y_pred[:, 0:FEATURE_COUNT]
  positive = y_pred[:, FEATURE_COUNT:2 * FEATURE_COUNT]
  return K.sqrt(K.sum(K.square(anchor - positive), axis=1) + K.epsilon())

def negative_distance(y_pred):
  anchor = y_pred[:, 0:FEATURE_COUNT]
  negative = y_pred[:, 2 * FEATURE_COUNT:3 * FEATURE_COUNT]
  return K.sqrt(K.sum(K.square(anchor - negative), axis=1) + K.epsilon())

def pmean(y_true, y_pred):
  return K.mean(positive_distance(y_pred))

def pvar(y_true, y_pred):
  return K.var(positive_distance(y_pred)) / pmean(y_true, y_pred)

def nmean(y_true, y_pred):
  return K.mean(negative_distance(y_pred))

def nvar(y_true, y_pred):
  return K.var(negative_distance(y_pred)) / nmean(y_true, y_pred)

def accuracy(y_true, y_pred):
  return K.mean(K.greater(
      negative_distance(y_pred) - positive_distance(y_pred),
      math.sqrt(MARGIN) / 2.0))

class JoinInputs(keras.layers.Layer):
  def call(self, inputs):
    if not isinstance(inputs, list):
      raise ValueError('`inputs` should be a list.')
    return K.concatenate([
      K.cast(K.expand_dims(inputs[1], axis=-1), 'float32'),
      K.one_hot(inputs[0], MAX_CHAR + 2),
    ])

  def compute_output_shape(self, input_shapes):
    if not isinstance(input_shapes, list):
      raise ValueError('`input_shapes` should be a list.')
    return input_shapes[0] + (MAX_CHAR + 3,)

  def compute_mask(self, inputs, masks=None):
    return K.not_equal(inputs[0], 0)

class NormalizeToSphere(keras.layers.Layer):
  def call(self, x):
    return K.l2_normalize(x + K.epsilon(), axis=1)

def create_siamese(input_shape):
  codes = Input(shape=input_shape, dtype='int32', name='codes')
  deltas = Input(shape=input_shape, name='deltas')

  joint_input = JoinInputs(name='join_inputs')([ codes, deltas ])

  x = GRU(128, name='gru',
          kernel_regularizer=L2, recurrent_regularizer=L2)(joint_input)

  # Residual layers (aka side-chain)
  sc = Dense(128, name='residual_l2', kernel_regularizer=L2,
             activation='relu')(x)
  sc = Dense(128, name='residual_l3', kernel_regularizer=L2,
             activation='relu')(sc)

  # Merge
  x = keras.layers.Add(name='residual_combine')([ x, sc ])

  # Residual layers (aka side-chain)
  sc = Dense(128, name='residual_l4', kernel_regularizer=L2,
             activation='relu')(x)
  sc = Dense(128, name='residual_l5', kernel_regularizer=L2,
             activation='relu')(sc)

  # Merge
  x = keras.layers.Add(name='residual_combine')([ x, sc ])

  x = Dense(FEATURE_COUNT, name='features', kernel_regularizer=L2)(x)

  output = NormalizeToSphere(name='normalize')(x)
  return Model(name='siamese', inputs=[ codes, deltas ], outputs=output)

def create_model(input_shape, siamese):
  anchor = {
    'codes': Input(shape=input_shape, dtype='int32', name='anchor_codes'),
    'deltas': Input(shape=input_shape, name='anchor_deltas')
  }
  positive = {
    'codes': Input(shape=input_shape, dtype='int32', name='positive_codes'),
    'deltas': Input(shape=input_shape, name='positive_deltas')
  }
  negative = {
    'codes': Input(shape=input_shape, dtype='int32', name='negative_codes'),
    'deltas': Input(shape=input_shape, name='negative_deltas')
  }

  anchor_activations = siamese([ anchor['codes'], anchor['deltas'] ])
  positive_activations = siamese([ positive['codes'], positive['deltas'] ])
  negative_activations = siamese([ negative['codes'], negative['deltas'] ])

  inputs = [
    anchor['codes'], anchor['deltas'],
    positive['codes'], positive['deltas'],
    negative['codes'], negative['deltas'],
  ]

  outputs = keras.layers.concatenate([
    anchor_activations,
    positive_activations,
    negative_activations
  ], axis=-1)

  return Model(name='triplet', inputs=inputs, outputs=outputs)

def create_regression(input_shape, siamese):
  codes = Input(shape=input_shape, dtype='int32', name='codes')
  deltas = Input(shape=input_shape, name='deltas')

  x = siamese([ codes, deltas ])

  # Reduce number of features
  x = Dense(len(dataset.LABELS), name='reduction', activation='softmax')(x)

  return Model(name='regression', inputs=[ codes, deltas ], outputs=x)

def create(sequence_len):
  input_shape = (sequence_len,)

  siamese = create_siamese(input_shape)
  model = create_model(input_shape, siamese)
  regression = create_regression(input_shape, siamese)

  return (siamese, model, regression)

def generate_dummy(triplets):
  return np.zeros([ triplets['anchor_codes'].shape[0], FEATURE_COUNT ])

def generate_one_hot_regression(indices):
  result = []
  for index in indices:
    one_hot = np.zeros([ len(dataset.LABELS) ], dtype='int32')
    one_hot[index] = 1
    result.append(one_hot)
  return np.array(result)

# Standard set of metrics
metrics = [
  pmean, pvar,
  nmean, nvar,
  accuracy
]
