#!/usr/bin/env npx ts-node

import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';

import { Dataset, Output } from '../src/dataset';

const DATASETS_DIR = path.join(__dirname, '..', 'datasets');
const MTURK_DIR = path.join(__dirname, '..', 'mturk', 'datasets');
const OUT_DIR = path.join(__dirname, '..', 'out');

const files: string[] = fs.readdirSync(MTURK_DIR)
  .filter((file) => /\.json$/.test(file))
.map((file) => file.replace(/\.json$/, ''));

const d = new Dataset();

files.forEach((name) => {
  const content = fs.readFileSync(path.join(MTURK_DIR, name + '.json'));
  const parsed = JSON.parse(content);

  // Skip fake datasets
  if (d.generate(parsed).length < 20) {
    return;
  }

  fs.writeFileSync(path.join(DATASETS_DIR,
    'sv-' + name.slice(0, 8) + '.json'), content);
});
