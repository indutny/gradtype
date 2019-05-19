'use strict';

const fs = require('fs');

const DATA = JSON.parse(fs.readFileSync(process.argv[2]).toString());
const OUT_FILE = process.argv[3];

function distance(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return 1 - dot;
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
  const features = flatten(byCategory(data));

  const res = Buffer.alloc(4 + features.length * features.length * 4);
  res.writeInt32LE(features.length, 0);

  for (const [ i, a ] of features.entries()) {
    for (let j = i + 1; j < features.length; j++) {
      const d = distance(a, features[j]);
      res.writeFloatLE(d, 4 * (1 + i * features.length + j));
      res.writeFloatLE(d, 4 * (1 + j * features.length + i));
    }

    res.writeFloatLE(0, 4 * (1 + i * features.length + i));
  }
  console.log(features.length);

  return res;
}

const matrices = {
  train: matrix(DATA.train),
  validate: matrix(DATA.validate),
};


fs.writeFileSync(OUT_FILE + '.train', matrices.train);
fs.writeFileSync(OUT_FILE + '.val', matrices.validate);
