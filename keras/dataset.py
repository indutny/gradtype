import math
import os
import numpy as np
import random
import struct
import time
import json

# Mini-batch size
TRIPLET_MINI_BATCH = 30

# Max attempts per dataset
TRIPLET_MAX_NEGATIVE_ATTEMPTS = 3

# Maximum character code
MAX_CHAR = 27

# Percent of validation data (it'll end up being more in the end, so be gentle)
VALIDATE_PERCENT = 0.1

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

  # Add some datasets that wouldn't be on the training list at all
  for ds in datasets[:ds_split_i]:
    # No need to add this to regression training
    if kind is 'triple':
      validate.append(ds)

  for ds in datasets[ds_split_i:]:
    split_i = int(math.floor(VALIDATE_PERCENT * len(ds)))
    train.append(ds[split_i:])
    validate.append(ds[0:split_i])

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

# Inspired by: https://arxiv.org/pdf/1503.03832.pdf
def gen_triplets(model, datasets):
  start = time.time()

  # Shuffle sequences in datasets first, and cut them
  mini_datasets = []
  for ds in datasets:
    np.random.shuffle(ds)
    mini_datasets.append(ds[:TRIPLET_MINI_BATCH])

  print('Shuffled {}'.format(time.time() - start))

  # Evaluate network on those mini batches
  features = evaluate_model(model, mini_datasets)
  print('Evaluated {}'.format(time.time() - start))

  anchor_list = []
  positive_list = []
  negative_list = []
  for i in range(0, len(mini_datasets)):
    anchor_ds = mini_datasets[i]
    anchor_ds_features = features[i]

    for j in range(0, len(anchor_ds)):
      anchor_features = anchor_ds_features[j]

      # Fully connected mini-batch
      for positive_index in range(j + 1, len(anchor_ds)):
        attempts = 0
        best_negative_index = 0
        best_negative_distance = float('inf')
        best_negative_ds = None
        while True:
          negative_ds_index = random.randrange(0, len(mini_datasets))
          if negative_ds_index == i:
            continue

          negative_ds_features = features[negative_ds_index]

          # Compute distances
          negative_ds_distances = negative_ds_features - anchor_features
          negative_ds_distances **= 2;
          negative_ds_distances = np.mean(negative_ds_distances, axis=-1)

          negative_index = np.argmin(negative_ds_distances)
          negative_distance = negative_ds_distances[negative_index]

          if negative_distance < best_negative_distance:
            best_negative_distance = negative_distance
            best_negative_index = negative_index
            best_negative_ds = mini_datasets[negative_ds_index]

          attempts += 1
          if attempts >= TRIPLET_MAX_NEGATIVE_ATTEMPTS:
            break

        # Now we have both positive and negative sequences - emit!
        anchor_list.append(anchor_ds[j])
        positive_list.append(anchor_ds[positive_index])
        negative_list.append(best_negative_ds[best_negative_index])

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
