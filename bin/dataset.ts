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
    dataset: () => d.generate(entry.data),
  };
});

function encodeLine(index: number, table: ReadonlyArray<number>): string {
  return [ index ].concat(table).join(',') + '\n';
}

function* group(kind: 'train' | 'validate'): Iterator<IGroup> {
  const anchorDS = datasets.map((d) => d.dataset());
  const positiveDS = datasets.map((d) => d.dataset());
  const negativeDS = datasets.map((d) => d.dataset());

  function isEqual(a: number[], b: number[]): boolean {
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) {
        return false;
      }
    }
    return true;
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
      if (positive.done || isEqual(anchor.value, positive.value)) {
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

function writeCSV(file: string, groups: Iterator<string>): void {
  const fd = fs.openSync(path.join(OUT_DIR, file), 'w');
  for (const entry of groups) {
    fs.writeSync(fd, entry.anchor);
    fs.writeSync(fd, '\n');
    fs.writeSync(fd, entry.positive);
    fs.writeSync(fd, '\n');
    fs.writeSync(fd, entry.negative);
    fs.writeSync(fd, '\n');
  }
  fs.closeSync(fd);
}

writeCSV('train.csv', group('train'));
writeCSV('validate.csv', group('validate'));
