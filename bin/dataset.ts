#!/usr/bin/env npx ts-node

import * as assert from 'assert';
import { Buffer } from 'buffer';
import * as fs from 'fs';
import * as path from 'path';

import { Dataset, Output, Sequence } from '../src/dataset';

const MIN_SEQUENCE_COUNT = 10;

let totalSequences = 0;
const totalSequenceLen = {
  mean: 0,
  min: Infinity,
  max: -Infinity,
};

const DATASETS_DIR = path.join(__dirname, '..', 'datasets');
const SENTENCES_FILE = path.join(DATASETS_DIR, 'sentences.json');
const OUT_DIR = path.join(__dirname, '..', 'out');

let sentences = JSON.parse(fs.readFileSync(SENTENCES_FILE).toString());
sentences = sentences.map((line: string) => line.toLowerCase());

const labels: string[] = fs.readdirSync(DATASETS_DIR)
  .filter((file) => /\.json$/.test(file))
  .map((file) => file.replace(/\.json$/, ''))
  .filter((file) => file !== 'index' && file !== 'sentences');

function encodeSequence(sequence: Sequence) {
  totalSequences++;

  totalSequenceLen.mean += sequence.length;
  totalSequenceLen.min = Math.min(totalSequenceLen.min, sequence.length);
  totalSequenceLen.max = Math.max(totalSequenceLen.max, sequence.length);

  const enc = Buffer.alloc(4 + sequence.length * 12);
  enc.writeUInt32LE(sequence.length, 0);

  let nonEmpty = false;
  for (let i = 0; i < sequence.length; i++) {
    const code = sequence[i].code;
    if (code !== -1) {
      nonEmpty = true;
    }
    enc.writeInt32LE(code, 4 + i * 12);
    enc.writeFloatLE(sequence[i].hold, 4 + i * 12 + 4);
    enc.writeFloatLE(sequence[i].duration, 4 + i * 12 + 8);
  }
  assert(nonEmpty);
  return enc;
}

let errors = 0;

let datasets = labels.map((name) => {
  const file = path.join(DATASETS_DIR, name + '.json');
  return {
    data: JSON.parse(fs.readFileSync(file).toString()),
    name,
  };
}).map((entry) => {
  const d = new Dataset(sentences);

  if (entry.data.version === 2) {
    const sequences = d.check(entry.name, entry.data.sequences);
    errors += entry.data.sequences.length - sequences.length;
    return {
      name: entry.name,
      sequences,
    };
  }

  return {
    name: entry.name,
    sequences: d.generate(entry.data),
  };
}).filter((entry) => {
  return entry.sequences.length > MIN_SEQUENCE_COUNT;
});

fs.writeFileSync(path.join(DATASETS_DIR, 'index.json'), JSON.stringify(
  datasets.map((d) => d.name), null, 2));


datasets.slice()
  .sort((a, b) => b.sequences.length - a.sequences.length)
  .forEach((ds) => console.log(ds.name, ds.sequences.length));

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

fs.writeFileSync(path.join(OUT_DIR, 'lstm.json'), JSON.stringify(datasets));

console.log('Total sequence count: %d', totalSequences);
console.log('Mean length: %s',
  (totalSequenceLen.mean / totalSequences).toFixed(2));
console.log('Min length: %s', totalSequenceLen.min.toFixed(2));
console.log('Max length: %s', totalSequenceLen.max.toFixed(2));
console.log('Errors: %d', errors);
