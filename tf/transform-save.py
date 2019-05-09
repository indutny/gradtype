import sys
import logging
import json

import numpy as np
import tensorflow as tf

# Internal
import dataset
from model import Model

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

SEED = 0x37255c25

import_path = sys.argv[1]
if import_path.endswith('.index'):
  import_path = import_path[:-6]
export_path = sys.argv[2]

model = Model(training=False)

input_shape = (None, dataset.MAX_SEQUENCE_LEN,)

p_codes = tf.placeholder(tf.int32, shape=input_shape, name='codes')
p_holds = tf.placeholder(tf.float32, shape=input_shape, name='holds')
p_deltas = tf.placeholder(tf.float32, shape=input_shape, name='deltas')

output = model.build(p_holds, p_codes, p_deltas)

with tf.Session() as sess:
  saver = tf.train.Saver()
  saver.restore(sess, import_path)

  out = {}
  for var in tf.trainable_variables():
    val = sess.run(var)
    out[var.name] = val.tolist()

  with open(export_path, 'w') as f:
    json.dump(out, f)

  exit(0)

  # Unused code
  inputs = {
    'input': tf.saved_model.utils.build_tensor_info(model.input),
    'state': tf.saved_model.utils.build_tensor_info(model.rnn_state),
  }

  outputs = {
    'action': tf.saved_model.utils.build_tensor_info(model.action),
  }

  signature_def = tf.saved_model.signature_def_utils.build_signature_def(
      inputs=inputs,
      outputs=outputs,
  )

  builder = tf.saved_model.builder.SavedModelBuilder(export_path)
  builder.add_meta_graph_and_variables(
      sess, [ tf.saved_model.tag_constants.SERVING ],
      signature_def_map={ 'compute_action': signature_def })

  builder.save()
