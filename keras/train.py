import math
import numpy as np
import os.path
import random
import struct

import keras.layers
import keras.preprocessing
from keras import backend as K
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM, \
  GaussianNoise

# This must match the constant in `src/dataset.ts`
MAX_CHAR=28
VALIDATE_PERCENT = 0.25

FEATURE_COUNT = 128

# Triple loss alpha
ALPHA = 0.1

TOTAL_EPOCHS = 50000
CONTINUOUS_EPOCHS = 50

#
# Input parsing below
#

def parse_datasets():
  datasets = []
  with open('./out/lstm.raw', 'rb') as f:
    dataset_count = struct.unpack('<i', f.read(4))[0]
    for i in range(0, dataset_count):
      sequence_count = struct.unpack('<i', f.read(4))[0]
      sequences = []
      for j in range(0, sequence_count):
        sequence_len = struct.unpack('<i', f.read(4))[0]
        sequence = []
        for k in range(0, sequence_len):
          row = np.zeros(MAX_CHAR + 1, dtype='float32')
          code = struct.unpack('<i', f.read(4))[0]
          row[code] = struct.unpack('f', f.read(4))[0]
          sequence.append(row)
        sequences.append(np.array(sequence, dtype='float32'))
      datasets.append(sequences)
  return datasets

def split_datasets(datasets):
  train = []
  validate = []
  for ds in datasets:
    split_i = int(math.floor(VALIDATE_PERCENT * len(ds)))
    train.append(np.array(ds[split_i:]))
    validate.append(np.array(ds[0:split_i]))
  return (train, validate)

print('Loading dataset')
datasets = parse_datasets()
sequence_len = len(datasets[0][0])
train_datasets, validate_datasets = split_datasets(datasets)

input_shape = (sequence_len, MAX_CHAR + 1)

def generate_triples(datasets):
  # TODO(indutny): use model to find better triples

  # Shuffle sequences in datasets first
  for ds in datasets:
    np.random.shuffle(ds)

  anchor_list = []
  positive_list = []
  negative_list = []
  for i in range(0, len(datasets) - 1):
    anchor_ds = datasets[i]
    negative_i = 0
    for j in range(0, len(anchor_ds) - 1, 2):
      anchor = anchor_ds[j]
      positive = anchor_ds[j + 1]
      negative_ds = datasets[random.randrange(i + 1, len(datasets))]
      if negative_i >= len(negative_ds):
        break
      negative = negative_ds[negative_i]
      negative_i += 1

      anchor_list.append(anchor)
      positive_list.append(positive)
      negative_list.append(negative)

  return {
    'anchor': np.array(anchor_list),
    'positive': np.array(positive_list),
    'negative': np.array(negative_list)
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

  model.add(GaussianNoise(0.1, input_shape=input_shape))
  model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5))

  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))

  model.add(Dense(FEATURE_COUNT, activation='linear'))
  model.add(NormalizeToSphere())

  return model

def create_model():
  siamese = create_siamese()

  anchor = Input(shape=input_shape, name='anchor')
  positive = Input(shape=input_shape, name='positive')
  negative = Input(shape=input_shape, name = 'negative')

  anchor_activations = siamese(anchor)
  positive_activations = siamese(positive)
  negative_activations = siamese(negative)

  merge = keras.layers.concatenate([
    anchor_activations,
    positive_activations,
    negative_activations
  ], axis=-1)

  return Model(inputs=[ anchor, positive, negative ], outputs=merge)

adam = Adam(lr=0.001)

model = create_model()
model.compile(adam, loss=triple_loss, metrics=[
  pmean, pvar,
  nmean, nvar
])

def generate_dummy(triples):
  return np.zeros([ triples['anchor'].shape[0], FEATURE_COUNT ])

for i in range(0, TOTAL_EPOCHS, CONTINUOUS_EPOCHS):
  callbacks = [
    TensorBoard(histogram_freq=10)
  ]

  print('Run #' + str(i))
  triples = generate_triples(train_datasets)
  val_triples = generate_triples(validate_datasets)
  model.fit(x=triples, y=generate_dummy(triples), batch_size=1024,
      initial_epoch=i,
      epochs=i + CONTINUOUS_EPOCHS,
      callbacks=callbacks,
      validation_data=(val_triples, generate_dummy(val_triples)))
  model.save('./out/gradtype-' + str(i) + '.h5')
