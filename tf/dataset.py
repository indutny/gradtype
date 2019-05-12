import os
import struct
import json

import numpy as np

# Maximum character code
MAX_CHAR = 28

# Sequence length
MAX_SEQUENCE_LEN = 64

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

def load_sequence(f, category, label):
  sequence_len = struct.unpack('<i', f.read(4))[0]

  codes = []
  holds = []
  deltas = []
  for k in range(0, sequence_len):
    code = struct.unpack('<i', f.read(4))[0]
    hold = struct.unpack('f', f.read(4))[0]
    delta = struct.unpack('f', f.read(4))[0]

    if code < -1 or code > MAX_CHAR:
      raise Exception('Invalid code: "{}"'.format(code))

    codes.append(code + 1)
    holds.append(hold)
    deltas.append(delta)
  codes = np.array(codes, dtype='int32')
  holds = np.array(holds, dtype='float32')
  deltas = np.array(deltas, dtype='float32')

  return {
    'category': category,
    'label': label,
    'codes': codes,
    'holds': holds,
    'deltas': deltas
  }

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

    for category in range(0, category_count):
      sequence_count = struct.unpack('<i', f.read(4))[0]
      sequences = []
      for j in range(0, sequence_count):
        sequences.append(load_sequence(f, category, labels[category]))
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
  return normalize_dataset(out)

def expand_sequence(seq, overlap):
  if overlap is None:
    overlap = MAX_SEQUENCE_LEN

  count = len(seq['codes'])

  # Pad
  if count < MAX_SEQUENCE_LEN:
    pad_size = MAX_SEQUENCE_LEN - len(seq['codes'])
    codes = seq['codes']
    holds = seq['holds']
    deltas = seq['deltas']

    codes = np.concatenate([ codes, np.zeros(pad_size, dtype='int32') ])
    holds = np.concatenate([ holds, np.zeros(pad_size, dtype='float32') ])
    deltas = np.concatenate([ deltas, np.zeros(pad_size, dtype='float32') ])

    padded_seq = seq.copy()
    padded_seq.update({
      'codes': codes,
      'holds': holds,
      'deltas': deltas,
      'sequence_len': count,
    })
    return [ padded_seq ]

  # Expand
  out = []
  for i in range(0, count - MAX_SEQUENCE_LEN + 1, overlap):
    codes = seq['codes'][i:i + MAX_SEQUENCE_LEN]
    holds = seq['holds'][i:i + MAX_SEQUENCE_LEN]
    deltas = seq['deltas'][i:i + MAX_SEQUENCE_LEN]
    copy = seq.copy()
    copy.update({
      'codes': codes,
      'holds': holds,
      'deltas': deltas,
      'sequence_len': MAX_SEQUENCE_LEN,
    })
    out.append(copy)
  return out

def normalize_dataset(dataset):
  out = []
  for category in dataset:
    new_category = []
    for seq in category:
      new_category.append(normalize_sequence(seq))
    out.append(new_category)
  return out

def normalize_sequence(seq):
  res = seq.copy()
  deltas = seq['deltas']
  res.update({ 'deltas': deltas })
  return res

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

def flatten_dataset(dataset, k=None, random_state=None):
  if k is None:
    k = len(dataset)

  rand_state = np.random.RandomState(seed=random_state)

  perm = rand_state.permutation(len(dataset))
  categories = [ dataset[i] for i in perm[:k] ]

  max_category = 0
  sequences = []
  for category in categories:
    for seq in category:
      # NOTE: lame
      max_category = max(max_category, seq['category'])
      sequences.append(seq)

  return sequences

def gen_adversarial(count):
  shape = [ count, MAX_SEQUENCE_LEN ]
  codes = np.random.random_integers(1, MAX_CHAR + 1, shape)
  holds = np.random.exponential(1.0, shape)
  deltas = np.random.exponential(1.0, shape)
  return {
    'codes': codes,
    'holds': holds,
    'deltas': deltas,
    'sequence_len': MAX_SEQUENCE_LEN,
  }

def gen_regression(sequences, batch_size=256, adversarial_count=None):
  perm = np.random.permutation(len(sequences))
  batches = []
  for i in range(0, len(perm), batch_size):
    batch = []
    batch_perm = perm[i:i + batch_size]

    categories = []
    codes = []
    holds = []
    deltas = []
    sequence_lens = []
    for j in batch_perm:
      seq = sequences[j]
      categories.append(seq['category'])
      codes.append(seq['codes'])
      holds.append(seq['holds'])
      deltas.append(seq['deltas'])
      sequence_lens.append(seq['sequence_len'])

    categories = np.array(categories)
    codes = np.array(codes)
    holds = np.array(holds)
    deltas = np.array(deltas)

    if adversarial_count != None:
      adversarial = gen_adversarial(adversarial_count)

      codes = np.concatenate([ codes, adversarial['codes'] ], axis=0)
      holds = np.concatenate([ holds, adversarial['holds'] ], axis=0)
      deltas = np.concatenate([ deltas, adversarial['deltas'] ], axis=0)
      sequence_lens = np.concatenate(
          [ sequence_lens, adversarial['sequence_len'] ], axis=0)

    batches.append({
      'categories': categories,
      'codes': codes,
      'holds': holds,
      'deltas': deltas,
      'sequence_lens': sequence_lens,
    })
  return batches
