#!/usr/bin/env npx ts-node

import { Buffer } from 'buffer';
import * as fs from 'fs';
import * as path from 'path';

import { Dataset, Output } from '../src/dataset';

const DATASETS_DIR = path.join(__dirname, '..', 'datasets');
const OUT_DIR = path.join(__dirname, '..', 'out');

const labels: string[] = require(path.join(DATASETS_DIR, 'index.json'));

function encodeSeq(sequence) {
  const enc = Buffer.alloc(4 + sequence.length * 8);
  enc.writeUInt32LE(sequence.length, 0);
  for (let i = 0; i < sequence.length; i++) {
    enc.writeUInt32LE(sequence[i].code, 4 + i * 8);
    enc.writeFloatLE(sequence[i].delta, 4 + i * 8 + 4);
  }
  return enc;
}

const datasets = labels.map((name) => {
  const file = path.join(DATASETS_DIR, name + '.json');
  return {
    data: JSON.parse(fs.readFileSync(file).toString()),
    name,
  };
}).map((entry) => {
  const d = new Dataset();

  return {
    name: entry.name,
    sequences: d.generate(entry.data).map((sequence) => encodeSeq(sequence)),
  };
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
    fs.writeSync(fd, seq);
  }
});
fs.closeSync(fd);
