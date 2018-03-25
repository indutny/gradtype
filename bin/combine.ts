#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as path from 'path';

import { Dataset } from '../src/dataset';

const fileLines = process.argv.slice(2).map((name, index) => {
  const lines = fs.readFileSync(name).toString().split(/\r\n|\r|\n/g)
    .map((line) => line.trim()).filter((line) => line);

  return lines.map((line) => line + ',' + index);
});

let min = Infinity;
for (const file of fileLines) {
  min = Math.min(min, file.length);
}

const csv = [];
for (const file of fileLines) {
  for (let i = 0; i < min; i++) {
    const index = Math.floor(Math.random() * file.length);
    const line = file[index];
    file.splice(index, 1);
    csv.push(line);
  }
}

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
