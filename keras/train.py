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
from keras.layers import Input, Dense, Dropout, BatchNormalization, GRU, \
  Embedding

# This must match the constant in `src/dataset.ts`
MAX_CHAR = 27
VALIDATE_PERCENT = 0.25

EMBEDDING_DIM = 8
FEATURE_COUNT = 128

# Triple loss alpha
ALPHA = 0.1

TOTAL_EPOCHS = 2000000

# Number of epochs before reshuffling triples
RESHUFFLE_EPOCHS = 50
SAVE_EPOCHS = 500

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
        codes = []
        deltas = []
        for k in range(0, sequence_len):
          code = struct.unpack('<i', f.read(4))[0]
          delta = struct.unpack('f', f.read(4))[0]

          if code < -1 or code > MAX_CHAR:
            print("Invalid code " + str(code))
            raise

          codes.append(code + 1)
          deltas.append(delta)
        codes = np.array(codes, dtype='int32')
        deltas = np.array(deltas, dtype='float32')
        sequences.append({ 'codes': codes, 'deltas': deltas })
      datasets.append(sequences)
  return datasets

def split_datasets(datasets):
  train = []
  validate = []
  for ds in datasets:
    split_i = int(math.floor(VALIDATE_PERCENT * len(ds)))
    train.append(ds[split_i:])
    validate.append(ds[0:split_i])
  return (train, validate)

print('Loading dataset')
datasets = parse_datasets()
sequence_len = len(datasets[0][0]['codes'])
train_datasets, validate_datasets = split_datasets(datasets)

input_shape = (sequence_len,)

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

  def get_codes(item_list):
    return np.array(list(map(lambda item: item['codes'], item_list)))
  def get_deltas(item_list):
    return np.array(list(map(lambda item: item['deltas'], item_list)))

  return {
    'anchor_codes': get_codes(anchor_list),
    'anchor_deltas': get_deltas(anchor_list),
    'positive_codes': get_codes(positive_list),
    'positive_deltas': get_deltas(positive_list),
    'negative_codes': get_codes(negative_list),
    'negative_deltas': get_deltas(negative_list),
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
  return K.var(positive_distance(y_pred)) / pmean(y_true, y_pred)

def nmean(y_true, y_pred):
  return K.mean(negative_distance(y_pred))

def nvar(y_true, y_pred):
  return K.var(negative_distance(y_pred)) / nmean(y_true, y_pred)

def accuracy(y_true, y_pred):
  return K.mean(K.greater(
      negative_distance2(y_pred) - positive_distance2(y_pred),
      0.0))

class JoinInputs(keras.layers.Layer):
  def call(self, inputs):
    if not isinstance(inputs, list):
      raise ValueError('`inputs` should be a list.')
    return K.concatenate([ inputs[0], K.expand_dims(inputs[1], axis=-1) ])

  def compute_output_shape(self, input_shapes):
    if not isinstance(input_shapes, list):
      raise ValueError('`input_shapes` should be a list.')
    first = input_shapes[0]
    return (first[0], first[1], first[2] + 1)

  def compute_mask(self, inputs, masks=None):
    if masks is None:
      return None
    if not isinstance(masks, list):
      raise ValueError('`masks` should be a list.')
    if not masks[1] is None:
      raise ValueError('`masks[1]` should be None.')
    return masks[0]

class NormalizeToSphere(keras.layers.Layer):
  def call(self, x):
    return K.l2_normalize(x, axis=1)

def create_siamese():
  codes = Input(shape=input_shape, dtype='int32', name='codes')
  deltas = Input(shape=input_shape, name='deltas')

  embedded_codes = Embedding(MAX_CHAR + 2, EMBEDDING_DIM, mask_zero=True)(codes)
  joint_input = JoinInputs()([ embedded_codes, deltas ])

  x = GRU(128, dropout=0.5, recurrent_dropout=0.5)(joint_input)
  x = Dense(128, name='l1', activation='relu')(x)

  x = Dropout(0.5)(x)
  x = Dense(FEATURE_COUNT, name='features')(x)
  output = NormalizeToSphere(name='normalize')(x)
  return Model(inputs=[ codes, deltas ], outputs=output)

def create_model():
  siamese = create_siamese()

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

  return Model(inputs=inputs, outputs=outputs)

adam = Adam(lr=0.001)

model = create_model()
model.compile(adam, loss=triple_loss, metrics=[
  pmean, pvar,
  nmean, nvar,
  accuracy
])

def generate_dummy(triples):
  return np.zeros([ triples['anchor_codes'].shape[0], FEATURE_COUNT ])

start_epoch = 0
for i in range(0, TOTAL_EPOCHS, RESHUFFLE_EPOCHS):
  try:
    model.load_weights('./out/gradtype-' + str(i) + '.h5')
  except IOError:
    break
  start_epoch = i

for i in range(start_epoch, TOTAL_EPOCHS, RESHUFFLE_EPOCHS):
  callbacks = [
    TensorBoard(histogram_freq=500, write_graph=False, embeddings_freq=500)
  ]

  triples = generate_triples(train_datasets)
  val_triples = generate_triples(validate_datasets)
  model.fit(x=triples, y=generate_dummy(triples), batch_size=256,
      initial_epoch=i,
      epochs=i + RESHUFFLE_EPOCHS,
      callbacks=callbacks,
      validation_data=(val_triples, generate_dummy(val_triples)))

  if i % SAVE_EPOCHS == 0:
    print("Saving...')
    model.save_weights('./out/gradtype-' + str(i + RESHUFFLE_EPOCHS) + '.h5')
