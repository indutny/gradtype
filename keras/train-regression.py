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
datasets, sequence_len = dataset.parse()
train_datasets, validate_datasets = dataset.split(datasets, 'regression')

train_x = dataset.gen_regression(train_datasets)
validate_x = dataset.gen_regression(validate_datasets)
train_y = gradtype_model.generate_one_hot_regression(train_x['labels'])
validate_y = gradtype_model.generate_one_hot_regression(validate_x['labels'])

#
# Load model
#

siamese, _, model = gradtype_model.create(sequence_len)
start_epoch = gradtype_utils.load_weights(model, 'gradtype-regr-')

adam = Adam(lr=0.001)

model.compile(adam, loss='categorical_crossentropy', metrics=[ 'accuracy' ])

#
# Train
#

tb = TensorBoard(histogram_freq=1000, write_graph=False)
for i in range(start_epoch, TOTAL_EPOCHS, SAVE_EPOCHS):
  callbacks = [ tb  ]
  end_epoch = i + SAVE_EPOCHS

  model.fit(x=train_x, y=train_y,
      batch_size=4096,
      initial_epoch=i,
      epochs=end_epoch,
      callbacks=callbacks,
      validation_data=(validate_x, validate_y))

  print("Saving...")
  fname_siamese = './out/gradtype-regr-siamese-{:08d}.h5'.format(end_epoch)
  fname_full = './out/gradtype-regr-full-{:08d}.h5'.format(end_epoch)
  siamese.save_weights(fname_siamese)
  model.save_weights(fname_full)
