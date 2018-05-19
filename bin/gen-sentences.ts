#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as path from 'path';

const SENTENCE_COUNT = 89;

const wilde = fs.readFileSync(path.join(__dirname, '..', 'data', 'wilde.txt'));

const text: string = wilde.toString().replace(/\s+/g, ' ');
const sentences = text.split(/\.+/g)
  .filter((line) => !/["?!]/.test(line))
  .map((line) => line.trim())
  .filter((line) => line.length > 32 && line.length < 64);

const res = [];
for (let i = 0; i < SENTENCE_COUNT; i++) {
  const index = Math.floor(sentences.length * Math.random());
  res.push(sentences[index]);
  sentences.splice(index, 1);
}

console.log(JSON.stringify(res, null, 2));
