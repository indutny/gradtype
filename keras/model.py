import math
import numpy as np

import keras.layers
from keras import backend as K
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, BatchNormalization, GRU, Activation, \
    Embedding, TimeDistributed, Conv1D, MaxPooling1D

# Internals
import dataset
from common import FEATURE_COUNT

EMBEDDING_SIZE = 3
GRU_MAJOR_SIZE = 64
GRU_MINOR_SIZE = 64
RESIDUAL_DEPTH = 0

# This must match the constant in `src/dataset.ts`
MAX_CHAR = dataset.MAX_CHAR

# Triplet loss margin
MARGIN = 0.2

# Saddle-point fix inverse steepness
STEEPNESS = 4.0

# Amount of kick to get out of saddle-point (must be positive)
KICK = 0.1

ACCURACY_PERCENT = 0.75

# Just a common regularizer
L2 = regularizers.l2(0.01)
RESIDUAL_L2 = regularizers.l2(0.001)

#
# Network configuration
#

def positive_distance2(y_pred):
  anchor = y_pred[:, 0:FEATURE_COUNT]
  positive = y_pred[:, FEATURE_COUNT:2 * FEATURE_COUNT]
  return K.sum(K.square(anchor - positive), axis=1)

def negative_distance2(y_pred):
  anchor = y_pred[:, 0:FEATURE_COUNT]
  negative = y_pred[:, 2 * FEATURE_COUNT:3 * FEATURE_COUNT]
  return K.sum(K.square(anchor - negative), axis=1)

def triplet_loss(y_true, y_pred):
  delta = positive_distance2(y_pred) - negative_distance2(y_pred)
  denom = 1.0 + KICK - K.exp(K.minimum(0.0, delta) / (STEEPNESS * MARGIN))

  # Slow down
  denom /= KICK
  return K.maximum(0.0, delta + MARGIN) / denom

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
  delta = negative_distance2(y_pred) - positive_distance2(y_pred)
  return K.mean(K.greater(delta, MARGIN * ACCURACY_PERCENT));

class JoinInputs(keras.layers.Layer):
  def call(self, inputs):
    if not isinstance(inputs, list):
      raise ValueError('`inputs` should be a list.')
    return K.concatenate([
      inputs[0],
      K.expand_dims(inputs[1], axis=-1),
    ])

  def compute_output_shape(self, input_shapes):
    if not isinstance(input_shapes, list):
      raise ValueError('`input_shapes` should be a list.')
    return input_shapes[1] + (EMBEDDING_SIZE + 1,)

class NormalizeToSphere(keras.layers.Layer):
  def call(self, x):
    return K.l2_normalize(x + K.epsilon(), axis=1)

def create_siamese(input_shape):
  codes = Input(shape=input_shape, dtype='int32', name='codes')
  deltas = Input(shape=input_shape, name='deltas')

  embedding = Embedding(MAX_CHAR + 2, EMBEDDING_SIZE, name='embed')(codes)
  joint_input = JoinInputs(name='join_inputs')([ embedding, deltas ])

  x = Conv1D(GRU_MAJOR_SIZE, 5, name='conv_input')(joint_input)
  x = MaxPooling1D(name='max_pool_input')(x)

  x = GRU(GRU_MAJOR_SIZE, name='gru_major', kernel_regularizer=L2,
          recurrent_dropout=0.3)(x)
  x = BatchNormalization(name='gru_minor_batch_norm')(x)

  for i in range(0, RESIDUAL_DEPTH):
    # Residual connection
    rc = Dense(32, name='rc{}_dense_minor'.format(i), activation='relu',
               kernel_regularizer=RESIDUAL_L2)(x)
    rc = Dense(GRU_MINOR_SIZE, name='rc{}_dense_major'.format(i),
               activation='relu', kernel_regularizer=RESIDUAL_L2)(rc)

    # Merge residual connection
    x = keras.layers.Add(name='rc{}_merge_add'.format(i))([ x, rc ])
    x = Activation('relu', name='rc{}_merge_relu'.format(i))(x)

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

  x = Activation('softmax')(x)
  return Model(name='regression', inputs=[ codes, deltas ], outputs=x)

def create(sequence_len):
  input_shape = (sequence_len,)

  siamese = create_siamese(input_shape)
  model = create_model(input_shape, siamese)
  regression = create_regression(input_shape, siamese)

  return (siamese, model, regression)

def generate_one_hot_regression(indices):
  result = []
  for index in indices:
    one_hot = np.zeros([ FEATURE_COUNT ], dtype='int32')
    one_hot[index] = 1
    result.append(one_hot)
  return np.array(result)

# Standard set of metrics
metrics = [
  pmean, pvar,
  nmean, nvar,
  accuracy
]
