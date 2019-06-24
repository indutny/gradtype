import os
import struct
import json

import numpy as np

# Maximum character code
MAX_CHAR = 28

# Sequence length
MAX_SEQUENCE_LEN = 57

# Percent of sequences in validation data
VALIDATE_PERCENT = 0.33

# Percent of categories in validation data (`triplet` mode only)
VALIDATE_CATEGORY_PERCENT = 0.5

# Seed for shuffling sequences in category before splitting into train/validate
VALIDATE_PERMUTATION_SEED = 0x6f3d755c

MAX_RANDOM_CUTOFF = 16

NORMALIZE = True

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

  # Normalize timing per-sample
  if NORMALIZE:
    mean_delta = np.mean(deltas)
    var_delta = np.sqrt(np.var(deltas))

    mean_hold = np.mean(holds)
    var_hold = np.sqrt(np.var(holds))

    deltas -= mean_delta
    deltas /= var_delta

    holds -= mean_hold
    holds /= var_hold

  return {
    'category': category,
    'label': label,
    'codes': codes,
    'holds': holds,
    'deltas': deltas,
    'sequence_len': len(codes),
  }

def load(mode='triplet'):
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
  return split(categories, mode)

def split(dataset, mode):
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
    'train': expand(train),
    'train_mask': train_mask,
    'validate': expand(validate),
    'validate_mask': validate_mask,
  }

def expand(dataset):
  out = []
  for category in dataset:
    out_category = []
    for seq in category:
      out_category.append(pad_or_trim_seq(seq))
    out.append(out_category)
  return normalize_dataset(out)

def pad_or_trim_seq(seq):
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
    })
    return padded_seq

  # Trim
  codes = seq['codes'][:MAX_SEQUENCE_LEN]
  holds = seq['holds'][:MAX_SEQUENCE_LEN]
  deltas = seq['deltas'][:MAX_SEQUENCE_LEN]
  copy = seq.copy()
  copy.update({
    'codes': codes,
    'holds': holds,
    'deltas': deltas,
    'sequence_len': min(seq['sequence_len'], MAX_SEQUENCE_LEN),
  })
  return copy

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

def randomize_seq(seq):
  sequence_len = seq['sequence_len']
  start_off = np.random.randint(min(sequence_len, MAX_RANDOM_CUTOFF))
  end_off = np.random.randint(
      min(sequence_len, MAX_RANDOM_CUTOFF) - start_off)
  sequence_len -= start_off + end_off

  if sequence_len <= 0:
    raise Exception('Negative sequence len')

  return pad_or_trim_seq({
    'category': seq['category'],
    'codes': seq['codes'][start_off:-end_off],
    'holds': seq['holds'][start_off:-end_off],
    'deltas': seq['deltas'][start_off:-end_off],
    'sequence_len': sequence_len,
  })

def shuffle_uniform(dataset, randomize=False):
  # Permutations within each category
  category_perm = [ [] for cat in dataset ]

  # If `True` - all elements from the category were emitted at least once
  complete = [ len(cat) == 0 for cat in dataset ]

  categories = []
  codes = []
  holds = []
  deltas = []
  sequence_lens = []
  while sum(complete) != len(dataset):
    # Permute category and iterate through each
    for category_i in np.random.permutation(len(dataset)):
      category = dataset[category_i]

      # In each category do random permutation
      perm = category_perm[category_i]
      if len(perm) == 0:
        perm = np.random.permutation(len(category))
        category_perm[category_i] = perm

      seq = category[perm[0]]
      if randomize:
        seq = randomize_seq(seq)
      yield seq
      perm = perm[1:]
      if len(perm) == 0:
        complete[category_i] = True

      category_perm[category_i] = perm

def gen_regression(dataset, batch_size, randomize=False):
  total = sum([ len(cat) for cat in dataset ])
  if batch_size is None:
    batch_size = total
  elif total % batch_size != 0:
    pad = batch_size - (total % batch_size)
    total += pad

  shuffle = shuffle_uniform(dataset, randomize=randomize)
  while True:
    batches = []
    for _ in range(0, total, batch_size):
      categories = []
      codes = []
      holds = []
      deltas = []
      sequence_lens = []

      for _ in range(0, batch_size):
        try:
          seq = next(shuffle)
        except StopIteration:
          shuffle = shuffle_uniform(dataset, randomize=randomize)
          seq = next(shuffle)

        categories.append(seq['category'])
        codes.append(seq['codes'])
        holds.append(seq['holds'])
        deltas.append(seq['deltas'])
        sequence_lens.append(seq['sequence_len'])

      categories = np.array(categories)
      codes = np.array(codes)
      holds = np.array(holds)
      deltas = np.array(deltas)

      batches.append({
        'categories': categories,
        'codes': codes,
        'holds': holds,
        'deltas': deltas,
        'sequence_lens': sequence_lens,
      })

    yield batches
