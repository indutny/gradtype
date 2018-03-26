#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as path from 'path';

import { Dataset, DatasetEntry } from '../src/dataset';

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

// Shuffle
for (let i = 0; i < csv.train.length - 1; i++) {
  const j = i + 1 + Math.floor(Math.random() * (csv.train.length - 1 - i));

  const t = csv.train[i];
  csv.train[i] = csv.train[j];
  csv.train[j] = t;
}

const OUT_DIR = path.join(__dirname, '..', 'out');

try {
  fs.mkdirSync(OUT_DIR);
} catch (e) {
  // no-op
}

function writeCSV(file: string, csv: ReadonlyArray<string>): void {
  const fd = fs.openSync(path.join(OUT_DIR, file), 'w');
  for (const line of csv) {
    fs.writeSync(fd, line);
    fs.writeSync(fd, '\n');
  }
  fs.closeSync(fd);
}

writeCSV('train.csv', csv.train);
writeCSV('validate.csv', csv.validate);
