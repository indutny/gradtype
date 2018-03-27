import numpy as np
import os.path
import struct

import keras.layers
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization

# [ prev char, next_char, normalized delta or one hot ]
INPUT_SHAPE=(29 * 29 * 2,)

FEATURE_COUNT = 128

# Triple loss alpha
ALPHA = 0.1

TOTAL_EPOCHS = 50000
CONTINUOUS_EPOCHS = 100

#
# Input parsing below
#

def parse_raw_single(name):
  with open(name, 'rb') as f:
    tables = []
    while True:
      word = f.read(4)
      if len(word) == 0:
        break
      size = struct.unpack('<i', word)[0]
      line = struct.unpack('f' * size, f.read(size * 4))
      line = np.array(line, dtype='float32')
      tables.append(line)
    return np.stack(tables)

def parse_dataset():
  np_file = './out/dataset.npy.npz'
  if os.path.isfile(np_file):
    raw = np.load(np_file)
  else:
    train = parse_raw_single('./out/train.raw')
    validate = parse_raw_single('./out/validate.raw')
    np.savez_compressed(np_file, train=train, validate=validate)
    raw = { 'train': train, 'validate': validate }

  train = {
    'anchor': raw['train'][0::3],
    'positive': raw['train'][1::3],
    'negative': raw['train'][2::3],
  }

  validate = {
    'anchor': raw['validate'][0::3],
    'positive': raw['validate'][1::3],
    'negative': raw['validate'][2::3],
  }

  return { 'train': train, 'validate': validate }

def create_dummy_y(subdataset):
  return np.zeros([ subdataset['anchor'].shape[0], FEATURE_COUNT ])

print('Loading dataset')
dataset = parse_dataset()
dummy_y = {
  'train': create_dummy_y(dataset['train']),
  'validate': create_dummy_y(dataset['validate']),
}

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

# Probably don't use these two in learning
def positive_distance(y_pred):
  return K.sqrt(positive_distance2(y_pred) + K.epsilon())

def negative_distance(y_pred):
  return K.sqrt(negative_distance2(y_pred) + K.epsilon())

def triple_loss(y_true, y_pred):
  return K.maximum(0.0,
      positive_distance2(y_pred) - negative_distance2(y_pred) + ALPHA)

def pmean(y_true, y_pred):
  return K.mean(positive_distance(y_pred))

def pvar(y_true, y_pred):
  return K.var(positive_distance(y_pred))

def nmean(y_true, y_pred):
  return K.mean(negative_distance(y_pred))

def nvar(y_true, y_pred):
  return K.var(negative_distance(y_pred))

class NormalizeToSphere(keras.layers.Layer):
  def call(self, x):
    norm = K.sqrt(K.sum(K.square(x), axis=1) + K.epsilon())
    return x / K.expand_dims(norm, 1)

  def compute_output_shape(self, input_shape):
    return input_shape

def create_siamese():
  model = Sequential()

  model.add(Dropout(0.2, input_shape=INPUT_SHAPE))

  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))

  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))

  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))

  model.add(Dense(FEATURE_COUNT, activation='linear'))
  model.add(NormalizeToSphere())

  return model

def create_model():
  siamese = create_siamese()

  anchor = Input(shape=INPUT_SHAPE, name='anchor')
  positive = Input(shape=INPUT_SHAPE, name='positive')
  negative = Input(shape=INPUT_SHAPE, name = 'negative')

  anchor_activations = siamese(anchor)
  positive_activations = siamese(positive)
  negative_activations = siamese(negative)

  merge = keras.layers.concatenate([
    anchor_activations,
    positive_activations,
    negative_activations
  ], axis=-1)

  return Model(inputs=[ anchor, positive, negative ], outputs=merge)

model = create_model()
model.compile('adam', loss=triple_loss, metrics=[
  pmean, pvar,
  nmean, nvar
])

for i in range(0, TOTAL_EPOCHS, CONTINUOUS_EPOCHS):
  print('Run #' + str(i))
  model.fit(x=dataset['train'], y=dummy_y['train'], batch_size=128,
      epochs=CONTINUOUS_EPOCHS,
      validation_data=(dataset['validate'], dummy_y['validate']))
  model.save('./out/gradtype-' + str(i) + '.h5')
