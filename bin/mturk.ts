#!/usr/bin/env npx ts-node

import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';

import { Dataset, Output } from '../src/dataset';

const DATASETS_DIR = path.join(__dirname, '..', 'datasets');
const MTURK_DIR = path.join(__dirname, '..', 'mturk', 'datasets');
const OUT_DIR = path.join(__dirname, '..', 'out');

const hashes = fs.readFileSync(process.argv[2]).toString().toLowerCase()
  .split(/[\n,"]/g).filter((word) => /^[a-z0-9]{64,64}$/.test(word));

const d = new Dataset();

hashes.forEach((hash) => {
  if (!fs.existsSync(path.join(MTURK_DIR, hash + '.json'))) {
    console.error('Not found: %j', hash);
    return false;
  }

  const content = fs.readFileSync(path.join(MTURK_DIR, hash + '.json'));
  const parsed = JSON.parse(content);

  // Skip fake datasets
  if (d.generate(parsed).length < 20) {
    console.error('Suspect: %j', hash);
    return;
  }

  fs.writeFileSync(path.join(DATASETS_DIR,
    'sv-' + hash.slice(0, 8) + '.json'), content);
  console.error('Success: %j', hash);
});
