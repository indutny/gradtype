import keras.layers
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

# Internals
import dataset
import model as gradtype_model
import utils as gradtype_utils

TOTAL_EPOCHS = 2000000

# Save weights every `SAVE_EPOCHS` epochs
SAVE_EPOCHS = 50

#
# Prepare dataset
#

print('Loading dataset')
datasets = dataset.parse()
skipgrams = dataset.skipgrams(datasets)

#
# Load model
#

encoder = gradtype_model.create_encoder()
autoencoder = gradtype_model.create_autoencoder(encoder)
start_epoch = gradtype_utils.load_weights(autoencoder, 'gradtype-skipgrams-')

adam = Adam(lr=0.001)

model.compile(adam, loss='categorical_crossentropy', metrics=[ 'accuracy' ])

#
# Train
#

tb = TensorBoard(histogram_freq=50,
                 log_dir=gradtype_utils.get_tensorboard_logdir())

callbacks = [ tb ]

for i in range(start_epoch, TOTAL_EPOCHS, SAVE_EPOCHS):
  end_epoch = i + SAVE_EPOCHS

  autoencoder.fit(x=skipgrams['x'], y=skipgrams['y'],
      batch_size=256,
      initial_epoch=i,
      epochs=end_epoch,
      callbacks=callbacks,
      validation_data=(validate_x, validate_y))

  print("Saving...")
  fname = './out/gradtype-skipgrams-{:08d}.h5'.format(end_epoch)
  siamese.save_weights(fname)
