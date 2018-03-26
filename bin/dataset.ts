#!/usr/bin/env npx ts-node

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
    dataset: d.generate(entry.data),
  };
});

function* group(kind: 'train' | 'validate'): Iterator<IGroup> {
  let added = true;
  while (added) {
    added = false;

    for (let i = 0; i < datasets.length; i++) {
      const entry = datasets[i];

      const first = entry.dataset[kind].next();
      if (first.done) {
        continue;
      }

      // Try to give half-the-same/half-the-other
      let j;
      if (Math.random() < 0.5) {
        j = i;
      } else {
        do {
          j = Math.floor(Math.random() * datasets.length);
        } while (j === i);
      }

      const second = datasets[j].dataset[kind].next();
      if (second.done) {
        continue;
      }

      yield {
        first: [ i ].concat(first.value).join(','),
        second: [ j ].concat(second.value).join(','),
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

function writeCSV(file: string, groups: Iterator<string>): void {
  const fd = fs.openSync(path.join(OUT_DIR, file), 'w');
  for (const entry of groups) {
    fs.writeSync(fd, entry.first);
    fs.writeSync(fd, '\n');
    fs.writeSync(fd, entry.second);
    fs.writeSync(fd, '\n');
  }
  fs.closeSync(fd);
}

writeCSV('train.csv', group('train'));
writeCSV('validate.csv', group('validate'));
