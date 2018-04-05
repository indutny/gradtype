import keras.layers
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

# Internals
import dataset
import model as gradtype_model
import utils as gradtype_utils

TOTAL_EPOCHS = 2000000

# Number of epochs before reshuffling triplets
RESHUFFLE_EPOCHS = 1

# Save weights every `SAVE_EPOCHS` epochs
SAVE_EPOCHS = 1

#
# Prepare dataset
#

print('Loading dataset')
datasets  = dataset.parse()
train_datasets, validate_datasets = dataset.split(datasets)

#
# Load model
#

siamese, model, _ = gradtype_model.create()
start_epoch = gradtype_utils.load_weights(siamese, 'gradtype-triplet-')

adam = Adam(lr=0.003)

model.compile(adam, loss=gradtype_model.triplet_loss,
              metrics=gradtype_model.metrics)

#
# Train
#

tb = TensorBoard(log_dir=gradtype_utils.get_tensorboard_logdir(),
    write_images=True)

callbacks = [ tb ]

if start_epoch == 0:
  print("Saving initial...")
  fname = './out/gradtype-triplet-{:08d}.h5'.format(start_epoch)
  siamese.save_weights(fname)

for i in range(start_epoch, TOTAL_EPOCHS, RESHUFFLE_EPOCHS):
  end_epoch = i + RESHUFFLE_EPOCHS

  train_gen = dataset.TripletGenerator('train', siamese, train_datasets,
      batch_size=32)
  validate_gen = dataset.TripletGenerator('validate', siamese,
      validate_datasets, batch_size=32)

  model.fit_generator(train_gen,
      initial_epoch=i,
      epochs=end_epoch,
      callbacks=callbacks,
      workers=8,
      shuffle=False,
      validation_data=validate_gen)

  if end_epoch % SAVE_EPOCHS == 0:
    print("Saving...")
    fname = './out/gradtype-triplet-{:08d}.h5'.format(end_epoch)
    siamese.save_weights(fname)
