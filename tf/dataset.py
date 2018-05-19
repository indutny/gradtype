import os
import struct
import json

import numpy as np

# Maximum character code
MAX_CHAR = 28

# Sequence length
MAX_SEQUENCE_LEN = 1024

# Percent of sequences in validation data
VALIDATE_PERCENT = 0.33

# Percent of categories in validation data (`triplet` mode only)
VALIDATE_CATEGORY_PERCENT = 0.5

# Seed for shuffling sequences in category before splitting into train/validate
VALIDATE_PERMUTATION_SEED = 0x6f3d755c

def load_labels():
  package_directory = os.path.dirname(os.path.abspath(__file__))
  index_json = os.path.join(package_directory, '..', 'datasets', 'index.json')
  with open(index_json, 'r') as f:
    return json.load(f)

def load_sequence(f):
  sequence_len = struct.unpack('<i', f.read(4))[0]

  rows = []
  for i in range(sequence_len):
    row = struct.unpack('B' * (MAX_CHAR + 1), f.read(MAX_CHAR + 1))
    rows.append(row)
  rows = np.array(rows, dtype='float32')
  return rows

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
        rows = load_sequence(f)

        sequences.append({
          'category': i,
          'label': labels[i],
          'rows': rows,
        })
      categories.append(sequences)
  return split(categories, mode, train_overlap, validate_overlap)

def split(dataset, mode, train_overlap, validate_overlap):
  rand_state = np.random.RandomState(seed=VALIDATE_PERMUTATION_SEED)
  category_perm = rand_state.permutation(len(dataset))
  if mode == 'triplet':
    train_cat_count = int(len(dataset) * (1.0 - VALIDATE_CATEGORY_PERCENT))
  else:
    # For now
    train_cat_count = len(dataset)

  train = []
  validate = []
  train_mask = [ False ] * len(dataset)
  validate_mask = [ False ] * len(dataset)

  for category_i in category_perm[:train_cat_count]:
    category = dataset[category_i]

    perm = rand_state.permutation(len(category))
    train_seq_count = int(len(category) * (1.0 - VALIDATE_PERCENT))
    train_mask[category_i] = True

    # Full categories only in triplet mode
    if mode == 'triplet':
      train.append(category)
      continue

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
      validate_mask[category_i] = True
      validate.append(dataset[category_i])

  return {
    'category_count': len(dataset),
    'train': expand(train, train_overlap),
    'train_mask': train_mask,
    'validate': expand(validate, validate_overlap),
    'validate_mask': validate_mask,
  }

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

  count = len(seq['rows'])

  # Pad
  if count < MAX_SEQUENCE_LEN:
    pad_size = MAX_SEQUENCE_LEN - count

    padding = np.zeros((pad_size, MAX_CHAR + 1,), dtype='float32')

    rows = np.concatenate([ seq['rows'], padding ])

    padded_seq = seq.copy()
    padded_seq.update({ 'rows': rows })
    return [ padded_seq ]

  # Expand
  out = []
  for i in range(0, count - MAX_SEQUENCE_LEN + 1, overlap):
    rows = seq['rows'][i:i + MAX_SEQUENCE_LEN]
    copy = seq.copy()
    copy.update({ 'rows': rows })
    out.append(copy)
  return out

def trim_dataset(dataset, batch_size=1, random_state=None):
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

  rand_state = np.random.RandomState(seed=random_state)

  out = []
  for category in dataset:
    if len(category) < min_len:
      print(category[0]['label'] + ' has not enough sequences')
      continue

    out_cat = []
    perm = rand_state.permutation(len(category))
    for i in perm[:min_len]:
      out_cat.append(category[i])
    out.append(out_cat)
  return out, min_len

def flatten_dataset(dataset, k=None):
  if k is None:
    k = len(dataset)

  perm = np.random.permutation(len(dataset))
  categories = [ dataset[i] for i in perm[:k] ]

  max_category = 0
  sequences = []
  for category in categories:
    for seq in category:
      # NOTE: lame
      max_category = max(max_category, seq['category'])
      sequences.append(seq)

  weights = np.zeros(max_category + 1, dtype='float32')
  for seq in sequences:
    weights[seq['category']] += 1.0

  min_count = np.min(np.where(weights > 0.0, weights, float('inf')))
  weights = np.where(weights > 0.0, min_count / (weights + 1e-12), 0.0)

  return sequences, weights

def gen_regression(sequences, batch_size=256):
  perm = np.random.permutation(len(sequences))
  batches = []
  for i in range(0, len(perm), batch_size):
    batch = []
    batch_perm = perm[i:i + batch_size]

    categories = []
    rows = []
    for j in batch_perm:
      seq = sequences[j]
      categories.append(seq['category'])
      rows.append(seq['rows'])

    categories = np.array(categories)
    rows = np.array(rows)

    batches.append({
      'categories': categories,
      'rows': rows,
    })
  return batches
