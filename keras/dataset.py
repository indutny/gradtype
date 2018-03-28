import math
import os
import numpy as np
import random
import struct
import json

package_directory = os.path.dirname(os.path.abspath(__file__))
index_json = os.path.join(package_directory, '..', 'datasets', 'index.json')
with open(index_json, 'r') as f:
  LABELS = json.load(f)

def parse(max_char):
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

          if code < -1 or code > max_char:
            print("Invalid code " + str(code))
            raise

          codes.append(code + 1)
          deltas.append(delta)
        codes = np.array(codes, dtype='int32')
        deltas = np.array(deltas, dtype='float32')
        sequences.append({ 'codes': codes, 'deltas': deltas })
      datasets.append(sequences)
  return datasets

def split(datasets, percent):
  train = []
  validate = []
  for ds in datasets:
    split_i = int(math.floor(percent * len(ds)))
    train.append(ds[split_i:])
    validate.append(ds[0:split_i])
  return (train, validate)

def apply_model(model, datasets):
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

def gen_triplets(model, datasets):
  # TODO(indutny): use model to find better triplets

  # Shuffle sequences in datasets first
  for ds in datasets:
    np.random.shuffle(ds)

  anchor_list = []
  positive_list = []
  negative_list = []
  for i in range(0, len(datasets) - 1):
    anchor_ds = datasets[i]
    negative_i = 0
    for j in range(0, len(anchor_ds) - 1, 2):
      anchor = anchor_ds[j]
      positive = anchor_ds[j + 1]
      negative_ds = datasets[random.randrange(i + 1, len(datasets))]
      if negative_i >= len(negative_ds):
        break
      negative = negative_ds[negative_i]
      negative_i += 1

      anchor_list.append(anchor)
      positive_list.append(positive)
      negative_list.append(negative)

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
