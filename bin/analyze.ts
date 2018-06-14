#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as path from 'path';

import { Dataset, Intermediate } from '../src/dataset';

const DATASETS_DIR = path.join(__dirname, '..', 'datasets');
const SENTENCES_FILE = path.join(DATASETS_DIR, 'sentences.json');

let sentences = JSON.parse(fs.readFileSync(SENTENCES_FILE).toString());
sentences = sentences.map((line) => line.toLowerCase());

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
  const d = new Dataset(sentences);

  const items = d.preprocess(entry.data);
  let buf: string[] = [];
  for (const item of items) {
    if (item === 'reset') {
      process.stdout.write(buf.join('\n') + '\n');
      buf = [];
      continue;
    } else if (item === 'invalid') {
      buf = [];
      continue;
    }
    buf.push([ item.code, item.delta ].join(','));
  }
});
