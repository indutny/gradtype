import keras.layers
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.metrics import top_k_categorical_accuracy

# Internals
import dataset
import model as gradtype_model
import utils as gradtype_utils

TOTAL_EPOCHS = 2000000

# Save weights every `SAVE_EPOCHS` epochs
SAVE_EPOCHS = 10

#
# Prepare dataset
#

print('Loading dataset')
datasets = dataset.parse()
train_datasets, validate_datasets = dataset.split(datasets, 'regression')

validate_x = dataset.gen_regression(validate_datasets)
validate_y = gradtype_model.generate_one_hot_regression(validate_x['labels'])

#
# Load model
#

siamese, _, model = gradtype_model.create()
start_epoch = gradtype_utils.load(siamese, 'gradtype-regr-')

adam = Adam(lr=0.01)

def top_5(y_true, y_pred):
  return top_k_categorical_accuracy(y_true, y_pred, k=5)

model.compile(adam, loss='categorical_crossentropy', metrics=[
  'accuracy', top_5 ])

#
# Train
#

tb = TensorBoard(histogram_freq=10,
                 log_dir=gradtype_utils.get_tensorboard_logdir())

callbacks = [ tb ]

for i in range(start_epoch, TOTAL_EPOCHS, SAVE_EPOCHS):
  end_epoch = i + SAVE_EPOCHS

  train_x = dataset.gen_regression(dataset.trim_dataset(train_datasets))
  train_y = gradtype_model.generate_one_hot_regression(train_x['labels'])

  model.fit(x=train_x, y=train_y,
      batch_size=4096,
      initial_epoch=i,
      epochs=end_epoch,
      callbacks=callbacks,
      validation_data=(validate_x, validate_y))

  print("Saving...")
  fname = './out/gradtype-regr-{:08d}.h5'.format(end_epoch)
  siamese.save_weights(fname)
