import math
import numpy as np

import keras.layers
from keras import backend as K
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, GRU, Activation, \
    Embedding, Reshape

# Internals
import dataset
from common import FEATURE_COUNT

EMBEDDING_SIZE = 7
GRU_SIZE = 256

# This must match the constant in `src/dataset.ts`
MAX_CHAR = dataset.MAX_CHAR

# Triplet loss margin
MARGIN = 0.2

ACCURACY_PERCENT = 0.75

# Just a common regularizer
L2 = regularizers.l2(0.001)

#
# Network configuration
#

def positive_distance(y_pred):
  anchor = y_pred[:, 0:FEATURE_COUNT]
  positive = y_pred[:, FEATURE_COUNT:2 * FEATURE_COUNT]
  return K.sqrt(K.sum(K.square(anchor - positive), axis=1) + K.epsilon())

def negative_distance(y_pred):
  anchor = y_pred[:, 0:FEATURE_COUNT]
  negative = y_pred[:, 2 * FEATURE_COUNT:3 * FEATURE_COUNT]
  return K.sqrt(K.sum(K.square(anchor - negative), axis=1) + K.epsilon())

def triplet_loss(y_true, y_pred):
  # Use non-squared distance as in https://arxiv.org/pdf/1703.07737.pdf to
  # prevent collapsing
  delta = positive_distance(y_pred) - negative_distance(y_pred)
  return K.softplus(delta)

def pmean(y_true, y_pred):
  return K.mean(positive_distance(y_pred))

def pvar(y_true, y_pred):
  return K.var(positive_distance(y_pred)) / pmean(y_true, y_pred)

def nmean(y_true, y_pred):
  return K.mean(negative_distance(y_pred))

def nvar(y_true, y_pred):
  return K.var(negative_distance(y_pred)) / nmean(y_true, y_pred)

def accuracy(y_true, y_pred):
  delta = negative_distance(y_pred) - positive_distance(y_pred)
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

def create_encoder():
  code = Input(shape=(1,), dtype='int32')
  prediction = Input(shape=(1,), dtype='int32')

  embedding = Embedding(MAX_CHAR + 2, EMBEDDING_SIZE, name='encoding')

  embedding_shape = (EMBEDDING_SIZE,)

  code_encoding = Reshape(embedding_shape)(embedding(code))
  prediction_encoding = Reshape(embedding_shape)(embedding(prediction))

  output = keras.layers.dot([ code_encoding, prediction_encoding ], 1)
  output = Activation('sigmoid')(output)

  return Model(inputs=[ code, prediction ], outputs=output)

def create_siamese(input_shape):
  codes = Input(shape=input_shape, dtype='int32', name='codes')
  deltas = Input(shape=input_shape, name='deltas')

  embedding = Embedding(MAX_CHAR + 2, EMBEDDING_SIZE, name='embed')(codes)
  joint_input = JoinInputs(name='join_inputs')([ embedding, deltas ])

  x = joint_input

  x = GRU(GRU_SIZE, name='gru', kernel_regularizer=L2,
          kernel_initializer='he_normal', recurrent_dropout=0.3)(x)

  # Spread activations uniformly over the sphere
  x = Dense(FEATURE_COUNT, name='features', kernel_regularizer=L2)(x)
  output = x
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

def create():
  input_shape = (dataset.SEQUENCE_LEN,)

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
