import os
import re

import keras.layers
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

# Internals
import dataset
import model as gradtype_model

TOTAL_EPOCHS = 2000000

# Number of epochs before reshuffling triplets
RESHUFFLE_EPOCHS = 50

# Save weights every `SAVE_EPOCHS` epochs
SAVE_EPOCHS = 50

#
# Input parsing below
#

print('Loading dataset')
datasets, sequence_len = dataset.parse()
train_datasets, validate_datasets = dataset.split(datasets)

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

siamese, model = gradtype_model.create(sequence_len)

for save in saved_epochs:
  try:
    model.load_weights(os.path.join('./out', save['name']))
  except IOError:
    continue
  start_epoch = save['epoch']
  print("Loaded weights from " + save['name'])
  break

adam = Adam(lr=0.00001)

model.compile(adam, loss=gradtype_model.triplet_loss,
              metrics=gradtype_model.metrics)

for i in range(start_epoch, TOTAL_EPOCHS, RESHUFFLE_EPOCHS):
  callbacks = [
    TensorBoard(histogram_freq=1000, write_graph=False)
  ]
  end_epoch = i + RESHUFFLE_EPOCHS

  triplets = dataset.gen_triplets(siamese, train_datasets)
  val_triplets = dataset.gen_triplets(siamese, validate_datasets)
  y = gradtype_model.generate_dummy(triplets)
  val_y = gradtype_model.generate_dummy(val_triplets)

  model.fit(x=triplets, y=y,
      batch_size=512,
      initial_epoch=i,
      epochs=end_epoch,
      callbacks=callbacks,
      validation_data=(val_triplets, val_y))

  if end_epoch % SAVE_EPOCHS == 0:
    print("Saving...")
    fname = './out/gradtype-{:08d}.h5'.format(end_epoch)
    model.save_weights(fname)
