#!/usr/bin/env npx ts-node

import * as assert from 'assert';
import { Buffer } from 'buffer';
import * as fs from 'fs';
import * as path from 'path';

import { Dataset, Output } from '../src/dataset';

let totalSequences = 0;

const DATASETS_DIR = path.join(__dirname, '..', 'datasets');
const OUT_DIR = path.join(__dirname, '..', 'out');

const labels: string[] = fs.readdirSync(DATASETS_DIR)
  .filter((file) => /\.json$/.test(file))
  .map((file) => file.replace(/\.json$/, ''))
  .filter((file) => file !== 'index');

fs.writeFileSync(path.join(DATASETS_DIR, 'index.json'), JSON.stringify(
  labels, null, 2));

function encodeSequence(sequence) {
  totalSequences++;

  const enc = Buffer.alloc(4 + sequence.length * 12);
  enc.writeUInt32LE(sequence.length, 0);

  let nonEmpty = false;
  for (let i = 0; i < sequence.length; i++) {
    const code = sequence[i].code;
    if (code !== -1) {
      nonEmpty = true;
    }
    enc.writeInt32LE(code, 4 + i * 12);
    enc.writeInt32LE(sequence[i].type, 4 + i * 12 + 4);
    enc.writeFloatLE(sequence[i].delta, 4 + i * 12 + 8);
  }
  assert(nonEmpty);
  return enc;
}

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
    sequences: d.generate(entry.data),
  };
});

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

console.log('Total sequence count: %d', totalSequences);
