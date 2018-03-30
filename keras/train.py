import keras.layers
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

# Internals
import dataset
import model as gradtype_model
import utils as gradtype_utils

TOTAL_EPOCHS = 2000000

# Number of epochs before reshuffling triplets
RESHUFFLE_EPOCHS = 50

# Save weights every `SAVE_EPOCHS` epochs
SAVE_EPOCHS = 50

#
# Prepare dataset
#

print('Loading dataset')
datasets, sequence_len = dataset.parse()
train_datasets, validate_datasets = dataset.split(datasets)

#
# Load model
#

siamese, model = gradtype_model.create(sequence_len)
start_epoch = gradtype_utils.load_weights(siamese, 'gradtype-triplet-')

adam = Adam(lr=0.00001)

model.compile(adam, loss=gradtype_model.triplet_loss,
              metrics=gradtype_model.metrics)

#
# Train
#

tb = TensorBoard(histogram_freq=1000, write_graph=False)

for i in range(start_epoch, TOTAL_EPOCHS, RESHUFFLE_EPOCHS):
  callbacks = [ tb ]
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
    fname = './out/gradtype-triplet-{:08d}.h5'.format(end_epoch)
    siamese.save_weights(fname)
