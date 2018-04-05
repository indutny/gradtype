#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as path from 'path';

import { Dataset, Intermediate } from '../src/dataset';

const DATASETS_DIR = path.join(__dirname, '..', 'datasets');

let labels = process.argv.slice(2);
if (labels.length === 0) {
  const index = fs.readFileSync(path.join(DATASETS_DIR, 'index.json'));
  labels = JSON.parse(index.toString());
}

const datasets = labels.map((name) => {
  const file = path.join(DATASETS_DIR, name + '.json');
  return {
    data: JSON.parse(fs.readFileSync(file).toString()),
    name,
  };
}).map((entry) => {
  const d = new Dataset();

  const items = d.preprocess(entry.data);
  for (const item of items) {
    if (item === 'reset') {
      continue;
    }
    console.log([ item.code, item.delta ].join(','));
  }
});
