import math
import os
import numpy as np
import random
import struct
import time
import json

# Mini-batch size
TRIPLET_MINI_BATCH = 16

# Maximum character code
MAX_CHAR = 27

# Percent of validation data (it'll end up being more in the end, so be gentle)
VALIDATE_PERCENT = 0.2

package_directory = os.path.dirname(os.path.abspath(__file__))
index_json = os.path.join(package_directory, '..', 'datasets', 'index.json')
with open(index_json, 'r') as f:
  LABELS = json.load(f)

def parse():
  datasets = []
  with open('./out/lstm.raw', 'rb') as f:
    dataset_count = struct.unpack('<i', f.read(4))[0]
    for i in range(0, dataset_count):
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
            print("Invalid code " + str(code))
            raise

          codes.append(code + 1)
          deltas.append(delta)
        codes = np.array(codes, dtype='int32')
        deltas = np.array(deltas, dtype='float32')
        sequences.append({ 'label': i, 'codes': codes, 'deltas': deltas })
      datasets.append(sequences)
  return datasets, sequence_len

def split(datasets, kind='triple'):
  train = []
  validate = []

  ds_split_i = int(math.floor(VALIDATE_PERCENT * len(datasets)))

  for ds in datasets[ds_split_i:]:
    split_i = int(math.floor(VALIDATE_PERCENT * len(ds)))
    train.append(ds[split_i:])
    validate.append(ds[0:split_i])

  # Add some datasets that wouldn't be on the training list at all
  for ds in datasets[:ds_split_i]:
    # No need to add this to regression training
    if kind is 'triple':
      validate.append(ds)

  return (train, validate)

def evaluate_model(model, datasets):
  slice_offsets = []
  codes = []
  deltas = []

  offset = 0
  for i in range(0, len(datasets)):
    start = offset
    ds = datasets[i]
    for seq in ds:
      codes.append(seq['codes'])
      deltas.append(seq['deltas'])
      offset += 1
    slice_offsets.append((start, offset))

  codes = np.array(codes)
  deltas = np.array(deltas)
  coordinates = model.predict(x={ 'codes': codes, 'deltas': deltas })
  result = []
  for offsets in slice_offsets:
    result.append(coordinates[offsets[0]:offsets[1]])
  return result

def gen_triplets_in_mini_batch(datasets, features):
  anchor_list = []
  positive_list = []
  negative_list = []

  for i in range(0, len(datasets)):
    anchor_ds = datasets[i]
    anchor_ds_features = features[i]

    negative_datasets = datasets[:i] + datasets[i + 1:]
    negative_features = features[:i] + features[i + 1:]

    # Form a circle of in `anchor_ds`
    for j in range(0, len(anchor_ds)):
      anchor = anchor_ds[j]
      anchor_features = anchor_ds_features[j]

      positive = anchor_ds[(j + 1) % len(anchor_ds)]
      positive_features = anchor_ds_features[(j + 1) % len(anchor_ds)]

      # Limit from https://arxiv.org/pdf/1503.03832.pdf (4)
      limit = anchor_features - positive_features
      limit **= 2
      limit = np.mean(limit, axis=-1)

      # Compute distances
      best_negative_per_ds = []
      for neg_feature in negative_features:
        distance = neg_feature - anchor_features
        distance **= 2;
        distance = np.mean(distance, axis=-1)
        distance = np.where(distance > limit, distance, float('inf'))
        best_index = np.argmin(distance, axis=-1)
        best_negative_per_ds.append(best_index)

      best_negative_ds = np.argmin(best_negative_per_ds, axis=-1)

      negative = negative_datasets[best_negative_ds]
      negative = negative[best_negative_per_ds[best_negative_ds]]

      # Now we have both positive and negative sequences - emit!
      anchor_list.append(anchor)
      positive_list.append(positive)
      negative_list.append(negative)

  return anchor_list, positive_list, negative_list

# Inspired by: https://arxiv.org/pdf/1503.03832.pdf
def gen_triplets(model, datasets):
  print('Generating triplets...')
  start = time.time()

  # Shuffle sequences in datasets first
  for ds in datasets:
    np.random.shuffle(ds)
  print('{} Shuffled'.format(time.time() - start))

  # Evaluate network on shuffled datasets
  features = evaluate_model(model, datasets)
  print('{} Evaluated'.format(time.time() - start))

  # Split into mini batches
  batch_slices = []
  for i in range(0, len(datasets)):
    ds = datasets[i]
    ds_features = features[i]

    slices = []
    for j in range(0, len(ds), TRIPLET_MINI_BATCH):
      slices.append({
        'dataset': ds[j:j + TRIPLET_MINI_BATCH],
        'features': ds_features[j:j + TRIPLET_MINI_BATCH],
      })
    batch_slices.append(slices)
  print('{} Split into mini batches'.format(time.time() - start))

  anchor_list = []
  positive_list = []
  negative_list = []

  while True:
    mini_batch = []
    mini_features = []
    for i in range(0, len(batch_slices)):
      b = batch_slices[i]
      if (len(b) == 0):
        continue
      mini_batch.append(b[0]['dataset'])
      mini_features.append(b[0]['features'])
      batch_slices[i] = b[1:]

    # No way to select negative
    if len(mini_batch) < 2:
      break
    print('{} Processing mini batch, len={}'.format(
          time.time() - start, len(mini_batch)))

    a, p, n = gen_triplets_in_mini_batch(mini_batch, mini_features)
    anchor_list += a
    positive_list += p
    negative_list += n

  def get_codes(item_list):
    return np.array(list(map(lambda item: item['codes'], item_list)))
  def get_deltas(item_list):
    return np.array(list(map(lambda item: item['deltas'], item_list)))

  print('Computed {}'.format(time.time() - start))

  return {
    'anchor_codes': get_codes(anchor_list),
    'anchor_deltas': get_deltas(anchor_list),
    'positive_codes': get_codes(positive_list),
    'positive_deltas': get_deltas(positive_list),
    'negative_codes': get_codes(negative_list),
    'negative_deltas': get_deltas(negative_list),
  }

def gen_regression(datasets):
  codes = []
  deltas = []
  labels = []

  for ds in datasets:
    for seq in ds:
      codes.append(seq['codes'])
      deltas.append(seq['deltas'])
      labels.append(seq['label'])

  codes = np.array(codes)
  deltas = np.array(deltas)

  return { 'codes': codes, 'deltas': deltas, 'labels': labels }
