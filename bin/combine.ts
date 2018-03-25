#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as path from 'path';

import { Dataset } from '../src/gradtype';

const csv = [];

process.argv.slice(2).map((name, index) => {
  const lines = fs.readFileSync(name).toString().split(/\r\n|\r|\n/g)
    .map((line) => line.trim()).filter((line) => line);

  lines.map((line) => line + ',' + index).forEach((line) => {
    csv.push(line);
  });
});

// Shuffle
for (let i = 0; i < csv.length; i++) {
  const j = i + Math.floor(Math.random() * (csv.length - i));

  const t = csv[i];
  csv[i] = csv[j];
  csv[j] = t;
}

csv.forEach((line) => {
  process.stdout.write(line + '\n');
});
