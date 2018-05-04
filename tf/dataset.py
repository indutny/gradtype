import os
import struct
import json

import numpy as np

# Maximum character code
MAX_CHAR = 27

# Sequence length
MAX_SEQUENCE_LEN = 32

# Percent of sequences in validation data
VALIDATE_PERCENT = 0.33

# Percent of categories in validation data (`triplet` mode only)
VALIDATE_CATEGORY_PERCENT = 0.25

# Seed for shuffling sequences in category before splitting into train/validate
VALIDATE_PERMUTATION_SEED = 0x6f3d755c

def load_labels():
  package_directory = os.path.dirname(os.path.abspath(__file__))
  index_json = os.path.join(package_directory, '..', 'datasets', 'index.json')
  with open(index_json, 'r') as f:
    return json.load(f)

def load(mode='triplet', overlap=None, train_overlap=None,
         validate_overlap = None):
  if overlap != None:
    train_overlap = overlap
    validate_overlap = overlap

  labels = load_labels()
  categories = []
  with open('./out/lstm.raw', 'rb') as f:
    category_count = struct.unpack('<i', f.read(4))[0]
    if category_count != len(labels):
      raise Exception("Invalid category count")

    for i in range(0, category_count):
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
            raise Exception('Invalid code: "{}"'.format(code))

          codes.append(code + 1)
          deltas.append(delta)
        codes = np.array(codes, dtype='int32')
        deltas = np.array(deltas, dtype='float32')
        sequences.append({
          'category': i,
          'label': labels[i],
          'codes': codes,
          'deltas': deltas
        })
      categories.append(sequences)
  return split(categories, mode, train_overlap, validate_overlap)

def split(dataset, mode, train_overlap, validate_overlap):
  rand_state = np.random.RandomState(seed=VALIDATE_PERMUTATION_SEED)
  category_perm = rand_state.permutation(len(dataset))
  train_cat_count = int(len(dataset) * (1.0 - VALIDATE_CATEGORY_PERCENT))

  train = []
  validate = []
  for category_i in category_perm[:train_cat_count]:
    category = dataset[category_i]

    perm = rand_state.permutation(len(category))
    train_seq_count = int(len(category) * (1.0 - VALIDATE_PERCENT))

    train_category = []
    for i in perm[:train_seq_count]:
      train_category.append(category[i])

    validate_category = []
    for i in perm[train_seq_count:]:
      validate_category.append(category[i])

    train.append(train_category)
    validate.append(validate_category)

  if mode == 'triplet':
    for category_i in category_perm[train_cat_count:]:
      validate.append(dataset[category_i])

  return expand(train, train_overlap), expand(validate, validate_overlap)

def expand(dataset, overlap):
  out = []
  for category in dataset:
    out_category = []
    for seq in category:
      out_category += expand_sequence(seq, overlap)
    out.append(out_category)
  return out

def expand_sequence(seq, overlap):
  if overlap is None:
    overlap = MAX_SEQUENCE_LEN

  count = len(seq['codes'])

  # Pad
  if count < MAX_SEQUENCE_LEN:
    pad_size = MAX_SEQUENCE_LEN - len(seq['codes'])
    codes = seq['codes']
    deltas = seq['deltas']

    codes = np.concatenate([ codes, np.zeros(pad_size, dtype='int32') ])
    deltas = np.concatenate([ deltas, np.zeros(pad_size, dtype='float32') ])

    padded_seq = seq.copy()
    padded_seq.update({ 'codes': codes, 'deltas': deltas })
    return [ padded_seq ]

  # Expand
  out = []
  for i in range(0, count - MAX_SEQUENCE_LEN + 1, overlap):
    codes = seq['codes'][i:i + MAX_SEQUENCE_LEN]
    deltas = seq['deltas'][i:i + MAX_SEQUENCE_LEN]
    copy = seq.copy()
    copy.update({ 'codes': codes, 'deltas': deltas })
    out.append(copy)
  return out

def trim_dataset(dataset, batch_size=1):
  min_len = None
  for category in dataset:
    if len(category) < batch_size:
      continue
    elif min_len == None:
      min_len = len(category)
    else:
      min_len = min(min_len, len(category))

  # Equal batches
  min_len -= min_len % batch_size

  out = []
  for category in dataset:
    if len(category) < min_len:
      print(category[0]['label'] + ' has not enough sequences')
      continue

    out_cat = []
    perm = np.random.permutation(len(category))
    for i in perm[:min_len]:
      out_cat.append(category[i])
    out.append(out_cat)
  return out, min_len

# TODO(indutny): use tf.data.Dataset
def gen_hard_batches(dataset, batch_size=32, k=None):
  if k is None:
    k = len(dataset)

  # Leave the same number of sequences in each batch
  dataset, sequence_count = trim_dataset(dataset, batch_size)

  perm = np.random.permutation(len(dataset))
  dataset = [ dataset[i] for i in perm[:k] ]

  batches = []
  for off in range(0, sequence_count, batch_size):
    batch = { 'codes': [], 'deltas': [] }
    for category in dataset:
      for seq in category[off:off + batch_size]:
        batch['codes'].append(seq['codes'])
        batch['deltas'].append(seq['deltas'])
    batch['codes'] = np.array(batch['codes'])
    batch['deltas'] = np.array(batch['deltas'])
    batches.append(batch)
  return batches

def flatten_dataset(dataset, k=None):
  if k is None:
    k = len(dataset)

  perm = np.random.permutation(len(dataset))
  categories = [ dataset[i] for i in perm[:k] ]

  sequences = []
  for category in categories:
    for seq in category:
      sequences.append(seq)
  return sequences

def gen_regression(sequences, batch_size=256):
  perm = np.random.permutation(len(sequences))
  batches = []
  for i in range(0, len(perm), batch_size):
    batch = []
    batch_perm = perm[i:i + batch_size]

    # Even batches
    if len(batch_perm) != batch_size:
      break

    categories = []
    codes = []
    deltas = []
    for j in batch_perm:
      seq = sequences[j]
      categories.append(seq['category'])
      codes.append(seq['codes'])
      deltas.append(seq['deltas'])

    categories = np.array(categories)
    codes = np.array(codes)
    deltas = np.array(deltas)

    batches.append({
      'categories': categories,
      'codes': codes,
      'deltas': deltas
    })
  return batches
