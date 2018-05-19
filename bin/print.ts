#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as path from 'path';

import { Dataset } from '../src/dataset';

const json = JSON.parse(fs.readFileSync(process.argv[2]).toString());

const d = new Dataset();

const items = d.preprocess(json);
for (const row of items) {
  if (row === 'reset') {
    console.log('------');
    continue;
  }
  console.log(row.join(''));
}
