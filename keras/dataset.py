import math
import os
import numpy as np
import random
import struct
import time
import json

from keras.utils import Sequence

# Internal
from common import FEATURE_COUNT

# Maximum character code
MAX_CHAR = 27

# Percent of validation data (it'll end up being more in the end, so be gentle)
VALIDATE_PERCENT = 0.2

# Sequence length
SEQUENCE_LEN = 40
OVERLAP = 1

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
  return datasets

def split(datasets, kind='triple'):
  train = []
  validate = []

  ds_split_i = int(math.floor(VALIDATE_PERCENT * len(datasets)))

  VALIDATION_FREQ = int(math.floor(1 / VALIDATE_PERCENT))
  for ds in datasets[ds_split_i:]:
    train_seqs = []
    validate_seqs = []
    for i in range(0, len(ds)):
      if i % VALIDATION_FREQ == 0:
        validate_seqs.append(ds[i])
      else:
        train_seqs.append(ds[i])
    train.append(train_seqs)
    validate.append(validate_seqs)

  # Add some datasets that wouldn't be on the training list at all
  for ds in datasets[:ds_split_i]:
    # No need to add this to regression training
    if kind is 'triple':
      validate.append(ds)

  return expand(train), expand(validate)

def expand(dataset):
  out_ds = []
  for group in dataset:
    out_group = []
    for seq in group:
      out_group += expand_sequence(seq)
    out_ds.append(out_group)
  return out_ds

def expand_sequence(seq):
  label = seq['label']
  count = len(seq['codes'])

  # Pad
  if count < SEQUENCE_LEN:
    pad_size = SEQUENCE_LEN - len(seq['codes'])
    return [ {
      'label': label,
      'codes': np.concatenate([
          seq['codes'],
          np.zeros(pad_size, dtype='int32') ]),
      'deltas': np.concatenate([
          seq['codes'],
          np.zeros(pad_size, dtype='float32') ])
    } ]

  # Expand
  out = []
  for i in range(0, count - SEQUENCE_LEN + 1):
    out.append({
      'label': label,
      'codes': seq['codes'][i:i + SEQUENCE_LEN],
      'deltas': seq['deltas'][i:i + SEQUENCE_LEN]
    })
  return out

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
class TripletGenerator(Sequence):
  def __init__(self, kind, model, datasets, batch_size = 32):
    self.kind = kind

    # Shuffle sequences in datasets first
    for ds in datasets:
      np.random.shuffle(ds)

    if kind == 'train':
      all_features = evaluate_model(model, datasets)
      augmented_datasets = []
      for (ds, ds_features) in zip(datasets, all_features):
        augmented_ds = []
        for (seq, features) in zip(ds, ds_features):
          single = { 'features': features }
          single.update(seq)
          augmented_ds.append(single)
        augmented_datasets.append(augmented_ds)
      datasets = augmented_datasets
    self.batches = self.build_batches(datasets, batch_size)

  def __len__(self):
    return len(self.batches)

  def __getitem__(self, i):
    batch = self.batches[i]
    positive_seqs = batch['positives']
    negative_seqs = batch['negatives']

    # We don't really care, it is fine that they're random
    if self.kind == 'validate':
      triplets = self.build_validate_triplets(positive_seqs, negative_seqs)
      x = triplets_to_x(triplets)
      return x, generate_dummy(x)

    negative_features = [ e['features'] for e in negative_seqs ]

    # Construct triplets
    triplets = { 'anchors': [], 'positives': [], 'negatives': [] }
    for i in range(0, len(positive_seqs)):
      anchor = positive_seqs[i]
      for j in range(i + 1, len(positive_seqs)):
        positive = positive_seqs[j]
        negative_i = self.find_best_negative(anchor['features'],
            positive['features'], negative_features)
        negative = negative_seqs[negative_i]

        triplets['anchors'].append(anchor)
        triplets['positives'].append(positive)
        triplets['negatives'].append(negative)

    x = triplets_to_x(triplets)
    return x, generate_dummy(x)

  def build_batches(self, datasets, batch_size):
    batches = []
    for i in range(0, len(datasets)):
      ds = datasets[i]
      negative_datasets = datasets[:i] + datasets[i + 1:]
      negative_seqs = np.concatenate(negative_datasets, axis=0)
      np.random.shuffle(negative_seqs)

      for j in range(0, len(ds), batch_size):
        batch = []

        # Fill half of the batch with positives
        positives = ds[j:j + batch_size]

        # Fill other half with negatives
        negatives = negative_seqs[:batch_size]
        negative_seqs = negative_seqs[batch_size:]

        if len(positives) != batch_size or len(negatives) != batch_size:
          break

        batches.append({
          'positives': positives,
          'negatives': negatives,
        })

    return batches

  def find_best_negative(self, anchor, positive, negative_features):
    limit = np.mean((anchor - positive) ** 2, axis=-1)
    distances = np.mean((anchor - negative_features) ** 2, axis=-1)
    distances = np.where(distances > limit, distances, float('inf'))

    # Kind of an assert
    if len(distances) == 0:
      return None
    return np.argmin(distances, axis=-1)

  def build_validate_triplets(self, positive_seqs, negative_seqs):
    triplets = { 'anchors': [], 'positives': [], 'negatives': [] }
    # Nothing to generate from
    if len(negative_seqs) == 0:
      return triplets

    for i in range(0, len(positive_seqs)):
      anchor = positive_seqs[i]
      j = 0
      for k in range(i + 1, len(positive_seqs)):
        positive = positive_seqs[k]
        negative = negative_seqs[j]

        triplets['anchors'].append(anchor)
        triplets['positives'].append(positive)
        triplets['negatives'].append(negative)

        j += 1
        if j == len(negative_seqs):
          break

    return triplets

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

def generate_dummy(triplets):
  return np.zeros([ triplets['anchor_codes'].shape[0], FEATURE_COUNT ])

# Internal

def triplets_to_x(triplets):
  def get_codes(item_list):
    return np.array(list(map(lambda item: item['codes'], item_list)))

  def get_deltas(item_list):
    return np.array(list(map(lambda item: item['deltas'], item_list)))

  return {
    'anchor_codes': get_codes(triplets['anchors']),
    'anchor_deltas': get_deltas(triplets['anchors']),
    'positive_codes': get_codes(triplets['positives']),
    'positive_deltas': get_deltas(triplets['positives']),
    'negative_codes': get_codes(triplets['negatives']),
    'negative_deltas': get_deltas(triplets['negatives']),
  }
