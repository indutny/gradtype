#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as path from 'path';

import { Dataset, Intermediate } from '../src/dataset';

const INPUT_DIR = process.argv[2];
const OUTPUT_DIR = process.argv[3];

const files = fs.readdirSync(INPUT_DIR);

files.filter((name) => {
  return name !== 'index.json';
}).map((name) => {
  return {
    data: JSON.parse(fs.readFileSync(path.join(INPUT_DIR, name)).toString()),
    name,
  };
}).map((entry) => {
  const d = new Dataset();

  const items = d.port(entry.data);
  const result = Array.from(items);

  fs.writeFileSync(path.join(OUTPUT_DIR, entry.name), JSON.stringify(result));
});
