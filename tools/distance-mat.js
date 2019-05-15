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

function meanDistance(left, right) {
  let sum = 0;
  for (const a of left) {
    const distances = [];
    for (const b of right) {
      distances.push(distance(a, b));
    }
    distances.sort();

    const count = Math.min(20, distances.length);
    let mean = 0;
    for (let i = 0; i < count; i++) {
      mean += distances[i];
    }
    mean /= count;

    sum += mean;
  }
  return sum / left.length;
}

function matrix(data) {
  const cats = Array.from(byCategory(data).values());

  const res = Buffer.alloc(4 + cats.length * cats.length * 4);
  res.writeInt32LE(cats.length, 0);

  for (const [ i, a ] of cats.entries()) {
    for (const [ j, b ] of cats.entries()) {
      const d = meanDistance(a, b);
      res.writeFloatLE(d, 4 * (1 + i * cats.length + j));
    }
  }

  return res;
}

const matrices = {
  train: matrix(DATA.train),
  validate: matrix(DATA.validate),
};


fs.writeFileSync(OUT_FILE + '.train', matrices.train);
fs.writeFileSync(OUT_FILE + '.val', matrices.validate);
