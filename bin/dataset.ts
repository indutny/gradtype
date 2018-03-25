#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as path from 'path';

import { Dataset } from '../src/dataset';

const events = JSON.parse(fs.readFileSync(process.argv[2]).toString());

const dataset = new Dataset();

const out = dataset.generate(events);

out.forEach((single) => {
  process.stdout.write(single.join(',') + '\n');
});
