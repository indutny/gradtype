#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as path from 'path';

import { Dataset, DatasetEntry } from '../src/dataset';

const datasets = process.argv.slice(2).map((name) => {
  const file = path.join(__dirname, '..', 'datasets', name + '.json');
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

const csv = {
  train: [],
  validate: [],
};

for (let i = 0; i < datasets.length; i++) {
  const entry = datasets[i];

  csv.train = csv.train.concat(
    entry.dataset.train.map((table) => [ i ].concat(table).join(',')));
  csv.validate = csv.validate.concat(
    entry.dataset.validate.map((table) => [ i ].concat(table).join(',')));
}

const OUT_DIR = path.join(__dirname, '..', 'out');

try {
  fs.mkdirSync(OUT_DIR);
} catch (e) {
  // no-op
}

const labels = JSON.stringify(datasets.map((d) => d.name));

fs.writeFileSync(path.join(OUT_DIR, 'train.csv'), csv.train.join('\n'));
fs.writeFileSync(path.join(OUT_DIR, 'validate.csv'), csv.validate.join('\n'));
fs.writeFileSync(path.join(OUT_DIR, 'labels.json'), labels);
