import keras.layers
import keras.utils
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.preprocessing.sequence import skipgrams
import numpy as np

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
full_sequence = dataset.gen_full_sequence(datasets)
sampling_table = dataset.gen_sampling_table(full_sequence)

#
# Load model
#

encoder = gradtype_model.create_encoder()
start_epoch = gradtype_utils.load_weights(encoder, 'gradtype-skipgrams-')

adam = Adam(lr=0.001)

encoder.compile(adam, loss='mse', metrics=[ 'accuracy' ])

#
# Train
#

tb = TensorBoard(log_dir=gradtype_utils.get_tensorboard_logdir())

callbacks = [ tb ]

for i in range(start_epoch, TOTAL_EPOCHS, SAVE_EPOCHS):
  end_epoch = i + SAVE_EPOCHS
  couples, labels = skipgrams(full_sequence, dataset.MAX_CHAR + 2,
      sampling_table=sampling_table)

  code_list = []
  prediction_list = []
  for (code, prediction) in couples:
    code_list.append(code)
    prediction_list.append(prediction)

  code_list = np.array(code_list)
  prediction_list = np.array(prediction_list)
  labels = np.array(labels)

  encoder.fit(x=[ code_list, prediction_list ], y=labels,
      batch_size=256,
      initial_epoch=i,
      epochs=end_epoch,
      callbacks=callbacks)

  print("Saving...")
  fname = './out/gradtype-skipgrams-{:08d}.h5'.format(end_epoch)
  encoder.save_weights(fname)
