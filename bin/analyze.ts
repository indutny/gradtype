#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as path from 'path';

import { Dataset, Intermediate } from '../src/dataset';

const datasets = process.argv.slice(2).map((name) => {
  const file = path.join(__dirname, '..', 'datasets', name + '.json');
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
    console.log([ item.fromCode, item.toCode, item.delta ].join(','));
  }
});
