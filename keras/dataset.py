import math
import os
import numpy as np
import random
import struct
import json

# Max attempts per candidate
TRIPLE_MAX_ATTEMPTS = 3

# Max attempts per dataset
TRIPLE_MAX_NEGATIVE_ATTEMPTS = 3

# Maximum character code
MAX_CHAR = 27

# Percent of validation data
VALIDATE_PERCENT = 0.25

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
        sequences.append({ 'codes': codes, 'deltas': deltas })
      datasets.append(sequences)
  return datasets, sequence_len

def split(datasets):
  train = []
  validate = []

  ds_split_i = int(math.floor(VALIDATE_PERCENT * len(datasets)))

  # Add some datasets that wouldn't be on the training list at all
  for ds in datasets[:ds_split_i]:
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

def best_triplet_candidate(kind, anchor_feature, target_features):
  best_distance = 0 if kind is 'positive' else float('inf')
  best_index = None
  for i in range(0, TRIPLE_MAX_ATTEMPTS):
    pick = random.randrange(0, len(target_features))

    distance = np.square(anchor_feature - target_features[pick]).sum(axis=-1)
    distance = np.sqrt(distance)

    # Find positives that are currently as far as possible
    if kind is 'positive':
      update = distance > best_distance

    # Find negatives that are as close as possible
    else:
      update = distance < best_distance

    if update:
      best_distance = distance
      best_index = pick
  return ( best_index, best_distance )

def gen_triplets(model, datasets):
  features = evaluate_model(model, datasets)

  # Shuffle sequences in datasets first
  for ds in datasets:
    np.random.shuffle(ds)

  # Shuffle dataset indicies
  dataset_indices = list(range(0, len(datasets)))
  np.random.shuffle(dataset_indices)

  # Keep the first half
  dataset_indices = dataset_indices[:int(len(dataset_indices) / 2)]

  anchor_list = []
  positive_list = []
  negative_list = []
  for i in dataset_indices:
    anchor_ds = datasets[i]
    anchor_ds_features = features[i]

    # Take anchors from first half, positives from the second
    half_anchor_ds = int(len(anchor_ds) / 2)
    for j in range(0, half_anchor_ds):
      anchor_features = anchor_ds_features[j]
      positive_index, positive_distance  = best_triplet_candidate('positive',
          anchor_features,
          anchor_ds_features[half_anchor_ds:])
      positive_index += half_anchor_ds

      attempts = 0
      best_negative_index = 0
      best_negative_distance = float('inf')
      best_negative_ds = None
      while True:
        negative_ds_index = random.randrange(0, len(datasets))
        if negative_ds_index == i:
          continue

        negative_ds_features = features[negative_ds_index]
        negative_index, negative_distance = best_triplet_candidate('negative',
            anchor_features, negative_ds_features)

        if negative_distance < best_negative_distance:
          best_negative_distance = negative_distance
          best_negative_index = negative_index
          best_negative_ds = datasets[negative_ds_index]

        attempts += 1
        if attempts >= TRIPLE_MAX_NEGATIVE_ATTEMPTS:
          break

      # Now we have both positive and negative sequences - emit!
      anchor_list.append(anchor_ds[j])
      positive_list.append(anchor_ds[positive_index])
      negative_list.append(best_negative_ds[best_negative_index])

  def get_codes(item_list):
    return np.array(list(map(lambda item: item['codes'], item_list)))
  def get_deltas(item_list):
    return np.array(list(map(lambda item: item['deltas'], item_list)))

  return {
    'anchor_codes': get_codes(anchor_list),
    'anchor_deltas': get_deltas(anchor_list),
    'positive_codes': get_codes(positive_list),
    'positive_deltas': get_deltas(positive_list),
    'negative_codes': get_codes(negative_list),
    'negative_deltas': get_deltas(negative_list),
  }
