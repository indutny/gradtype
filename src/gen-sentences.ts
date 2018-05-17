#!/usr/bin/env npx ts-node

const ALPHA = 'abcdefghijklmnopqrstuvwxyz ,.';
console.log(ALPHA.length);

const SENTENCE_LENGTH = 32;
const SENTENCE_COUNT = 200;

const res = [];

for (let i = 0; i < SENTENCE_COUNT; i++) {
  let s: string = '';
  for (let j = 0; j < SENTENCE_LENGTH; j++) {
    s += ALPHA[Math.floor(Math.random() * ALPHA.length)];
  }
  res.push(s);
}

console.log(JSON.stringify(res, null, 2));
