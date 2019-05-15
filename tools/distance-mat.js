'use strict';

const fs = require('fs');

const DATA = JSON.parse(fs.readFileSync(process.argv[2]).toString());
const OUT_FILE = process.argv[3];

function distance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] - b[i]) ** 2;
  }
  return Math.sqrt(sum);
}

function byCategory(data) {
  const res = new Map();

  for (const elem of data) {
    const label = elem.label || elem.category;
    if (!res.has(label)) {
      res.set(label, []);
    }

    res.get(label).push(elem.features);
  }

  return res;
}

function flatten(cat) {
  const out = [];
  for (const featureList of cat.values()) {
    out.push(...featureList);
  }
  return out;
}

function matrix(data) {
  const list = flatten(byCategory(data));

  const res = Buffer.alloc(4 + list.length * list.length * 4);
  res.writeInt32LE(list.length, 0);

  for (let i = 0; i < list.length; i++) {
    const a = list[i];
    for (let j = i + 1; j < list.length; j++) {
      const b = list[j];

      const d = distance(a, b);
      res.writeFloatLE(d, 4 * (1 + i * list.length + j));
      res.writeFloatLE(d, 4 * (1 + j * list.length + i));
    }
    res.writeFloatLE(0, 4 * (1 + i * list.length + i));
  }

  return res;
}

const matrices = {
  train: matrix(DATA.train),
  validate: matrix(DATA.validate),
};


fs.writeFileSync(OUT_FILE + '.train', matrices.train);
fs.writeFileSync(OUT_FILE + '.val', matrices.validate);
