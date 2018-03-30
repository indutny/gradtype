#!/usr/bin/env npx ts-node

import * as assert from 'assert';
import { Buffer } from 'buffer';
import * as fs from 'fs';
import * as path from 'path';

import { Dataset, Output } from '../src/dataset';

let totalSequences = 0;

const MAX_SEQUENCE_LEN = 2;

const DATASETS_DIR = path.join(__dirname, '..', 'datasets');
const OUT_DIR = path.join(__dirname, '..', 'out');

const labels: string[] = require(path.join(DATASETS_DIR, 'index.json'));

function encodeSequence(sequence) {
  totalSequences++;

  const enc = Buffer.alloc(4 + sequence.length * 8);
  enc.writeUInt32LE(sequence.length, 0);
  let nonEmpty = false;
  for (let i = 0; i < sequence.length; i++) {
    if (sequence[i].code !== -1)
      nonEmpty = true;
    enc.writeInt32LE(sequence[i].code, 4 + i * 8);
    enc.writeFloatLE(sequence[i].delta, 4 + i * 8 + 4);
  }
  assert(nonEmpty);
  return enc;
}

let mean = 0;
let count = 0;

let datasets = labels.map((name) => {
  const file = path.join(DATASETS_DIR, name + '.json');
  return {
    data: JSON.parse(fs.readFileSync(file).toString()),
    name,
  };
}).map((entry) => {
  const d = new Dataset();

  return {
    name: entry.name,
    sequences: d.generate(entry.data).map((sequence) => {
      mean += sequence.length;
      count++;

      return sequence;
    }),
  };
});


// Expand
mean /= count;
mean = Math.ceil(mean);
datasets = datasets.map((ds) => {
  const sequences = [];
  const max = Math.min(mean, MAX_SEQUENCE_LEN);

  function pad(sequence, length) {
    if (sequence.length > length) {
      return sequence.slice(0, length);
    } else if (sequence.length < length) {
      return sequence.concat(new Array(length - sequence.length).fill({
        code: -1,
        delta: 0,
      }));
    } else {
      return sequence;
    }
  }

  ds.sequences.forEach((seq) => {
    if (seq.length <= max) {
      sequences.push(seq);
      return;
    }

    for (let i = 0; i < seq.length - max; i++) {
      const slice = seq.slice(i, i + max);
      assert.strictEqual(slice.length, max);
      sequences.push(seq.slice(i, i + max));
    }
  });

  return {
    name: ds.name,
    sequences: sequences.map((seq) => pad(seq, max)),
  }
});

try {
  fs.mkdirSync(OUT_DIR);
} catch (e) {
  // no-op
}

const fd = fs.openSync(path.join(OUT_DIR, 'lstm.raw'), 'w');

const datasetCount = Buffer.alloc(4);
datasetCount.writeUInt32LE(datasets.length, 0);
fs.writeSync(fd, datasetCount);

datasets.forEach((ds) => {
  const count = Buffer.alloc(4);
  count.writeUInt32LE(ds.sequences.length, 0);
  fs.writeSync(fd, count);

  for (const seq of ds.sequences) {
    fs.writeSync(fd, encodeSequence(seq));
  }
});
fs.closeSync(fd);

console.log('Total sequence count: %d', totalSequences);
