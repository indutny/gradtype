import numpy as np
import os
import re

import keras.layers
import keras.preprocessing
from keras import backend as K
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Dropout, BatchNormalization, GRU

# Internals
import dataset
import visualize

# This must match the constant in `src/dataset.ts`
MAX_CHAR = 27
VALIDATE_PERCENT = 0.25

FEATURE_COUNT = 128

# Triple loss alpha
ALPHA = 0.1

TOTAL_EPOCHS = 2000000

# Number of epochs before reshuffling triplets
RESHUFFLE_EPOCHS = 50

# Save weights every `SAVE_EPOCHS` epochs
SAVE_EPOCHS = 500

# Number of epochs before generating image
VISUALIZE_EPOCHS = 250

#
# Input parsing below
#

print('Loading dataset')
datasets = dataset.parse(MAX_CHAR)
sequence_len = len(datasets[0][0]['codes'])
train_datasets, validate_datasets = dataset.split(datasets, VALIDATE_PERCENT)

input_shape = (sequence_len,)

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

# Loss function from https://arxiv.org/pdf/1611.05301.pdf
# See: https://arxiv.org/pdf/1412.6622.pdf
def triplet_loss(y_true, y_pred):
  return K.maximum(0.0,
      positive_distance2(y_pred) - negative_distance2(y_pred) + ALPHA)

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
      0.0))

class JoinInputs(keras.layers.Layer):
  def call(self, inputs):
    if not isinstance(inputs, list):
      raise ValueError('`inputs` should be a list.')
    return K.concatenate([
      K.one_hot(inputs[0], MAX_CHAR + 2),
      K.expand_dims(inputs[1], axis=-1)
    ])

  def compute_output_shape(self, input_shapes):
    if not isinstance(input_shapes, list):
      raise ValueError('`input_shapes` should be a list.')
    return input_shapes[0] + (MAX_CHAR + 3,)

  def compute_mask(self, inputs, masks=None):
    return K.not_equal(inputs[0], 0)

class NormalizeToSphere(keras.layers.Layer):
  def call(self, x):
    return K.l2_normalize(x, axis=1)

def create_siamese():
  codes = Input(shape=input_shape, dtype='int32', name='codes')
  deltas = Input(shape=input_shape, name='deltas')

  joint_input = JoinInputs()([ codes, deltas ])

  x = GRU(128, dropout=0.2, recurrent_dropout=0.2)(joint_input)

  x = Dropout(0.5)(x)
  x = Dense(FEATURE_COUNT, name='features')(x)
  output = NormalizeToSphere(name='normalize')(x)
  return Model(inputs=[ codes, deltas ], outputs=output)

siamese = create_siamese()

def create_model():
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
model.compile(adam, loss=triplet_loss, metrics=[
  pmean, pvar,
  nmean, nvar,
  accuracy
])

def generate_dummy(triplets):
  return np.zeros([ triplets['anchor_codes'].shape[0], FEATURE_COUNT ])

start_epoch = 0

weight_files = [ name for name in os.listdir('./out') if name.endswith('.h5') ]

saved_epochs = []
weight_file_re = re.compile(r"^gradtype-(\d+)\.h5$")
for name in weight_files:
  match = weight_file_re.match(name)
  if match == None:
    continue
  saved_epochs.append({ 'name': name, 'epoch': int(match.group(1)) })
saved_epochs.sort(key=lambda entry: entry['epoch'], reverse=True)

for save in saved_epochs:
  try:
    model.load_weights(os.path.join('./out', save['name']))
  except IOError:
    continue
  start_epoch = save['epoch']
  print("Loaded weights from " + save['name'])
  break

print("Visualizing initial PCA...")
visualize.pca(siamese, datasets, start_epoch)

for i in range(start_epoch, TOTAL_EPOCHS, RESHUFFLE_EPOCHS):
  callbacks = [
    TensorBoard(histogram_freq=2000, write_graph=False)
  ]
  end_epoch = i + RESHUFFLE_EPOCHS

  triplets = dataset.gen_triplets(siamese, train_datasets)
  val_triplets = dataset.gen_triplets(siamese, validate_datasets)
  model.fit(x=triplets, y=generate_dummy(triplets), batch_size=256,
      initial_epoch=i,
      epochs=end_epoch,
      callbacks=callbacks,
      validation_data=(val_triplets, generate_dummy(val_triplets)))

  if end_epoch % SAVE_EPOCHS == 0:
    print("Saving...")
    fname = './out/gradtype-{:06d}.h5'.format(end_epoch)
    model.save_weights(fname)

  if end_epoch % VISUALIZE_EPOCHS == 0:
    print("Visualizing PCA...")
    visualize.pca(siamese, datasets, end_epoch)
