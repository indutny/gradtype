#!/usr/bin/env npx ts-node

import { Buffer } from 'buffer';
import * as fs from 'fs';
import * as path from 'path';

import { Dataset, DatasetEntry } from '../src/dataset';

interface IGroup {
  readonly first: string;
  readonly second: string;
}

const DATASETS_DIR = path.join(__dirname, '..', 'datasets');

const labels: string[] = require(path.join(DATASETS_DIR, 'index.json'));

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
    dataset: () => d.generate(entry.data),
  };
});

function encodeLine(index: number, table: ReadonlyArray<number>): Buffer {
  const res = Buffer.alloc(4 + table.length * 4);
  res.writeUInt32LE(table.length, 0);
  for (let i = 0; i < table.length; i++) {
    res.writeFloatLE(table[i], 4 + i * 4);
  }
  return res;
}

function* group(kind: 'train' | 'validate'): Iterator<IGroup> {
  const anchorDS = datasets.map((d) => d.dataset());
  const positiveDS = datasets.map((d) => d.dataset());
  const negativeDS = datasets.map((d) => d.dataset());

  function isSimilar(a: number[], b: number[]): boolean {
    let distance = 0;
    for (let i = 0; i < a.length; i++) {
      distance += Math.pow(a[i] - b[i], 2);
    }
    distance /= a.length;
    distance = Math.sqrt(distance);
    return distance < 0.1;
  }

  let added = true;
  while (added) {
    added = false;

    for (let i = 0; i < anchorDS.length; i++) {
      const anchor = anchorDS[i][kind].next();
      if (anchor.done) {
        continue;
      }

      const positive = positiveDS[i][kind].next();
      if (positive.done || isSimilar(anchor.value, positive.value)) {
        continue;
      }

      let j = 0;
      do {
        j = Math.floor(Math.random() * negativeDS.length);
      } while (j === i);

      const negative = negativeDS[j][kind].next();
      if (negative.done) {
        continue;
      }

      yield {
        anchor: encodeLine(i, anchor.value),
        positive: encodeLine(i, positive.value),
        negative: encodeLine(j, negative.value),
      };
      added = true;
    }
  }
}

const OUT_DIR = path.join(__dirname, '..', 'out');

try {
  fs.mkdirSync(OUT_DIR);
} catch (e) {
  // no-op
}

// Remove Keras cache
try {
  fs.unlinkSync(path.join(OUT_DIR, 'dataset.npy.npz'));
} catch (e) {
  // no-op
}

function writeDataset(file: string, groups: Iterator<string>): void {
  const fd = fs.openSync(path.join(OUT_DIR, file), 'w');
  for (const entry of groups) {
    fs.writeSync(fd, entry.anchor);
    fs.writeSync(fd, entry.positive);
    fs.writeSync(fd, entry.negative);
  }
  fs.closeSync(fd);
}

writeDataset('train.raw', group('train'));
writeDataset('validate.raw', group('validate'));
