'use strict';

const fs = require('fs');

const DATA = JSON.parse(fs.readFileSync(process.argv[2]).toString());
const OUT_DISTANCE = process.argv[3];

const COSINE = true;

function cosine(a, b) {
  let dot = 0;
  let aNorm = 0;
  let bNorm = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    aNorm += a[i] ** 2;
    bNorm += b[i] ** 2;
  }
  return 1 - dot / Math.sqrt(aNorm) / Math.sqrt(bNorm);
}

function euclidian(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] - b[i]) ** 2;
  }
  return Math.sqrt(sum);
}

const distance = COSINE ? cosine : euclidian;

function crossDistance(data) {
  const max = data.length * (data.length - 1) / 2;
  let positives = new Float32Array(max);
  let negatives = new Float32Array(max);

  let positiveCount = 0;
  let negativeCount = 0;

  for (let i = 0; i < data.length; i++) {
    const category = data[i].category;
    const features = data[i].features;

    for (let j = i + 1; j < data.length; j++) {
      const otherCategory = data[j].category;
      const otherFeatures = data[j].features;

      const d = distance(features, otherFeatures);
      if (category === otherCategory) {
        positives[positiveCount++] = d;
      } else {
        negatives[negativeCount++] = d;
      }
    }
  }

  positives = positives.slice(0, positiveCount);
  negatives = negatives.slice(0, negativeCount);
  console.log('Got %d positives and %d negatives', positives.length,
    negatives.length);

  positives.sort();
  negatives.sort();

  return { positives, negatives };
}

console.log('step', DATA.step);

function encodeList(list) {
  const out = Buffer.alloc(4 + list.length * 4);
  let off = 0;
  out.writeInt32LE(list.length, off);
  off += 4;

  for (const val of list) {
    out.writeFloatLE(val, off);
    off += 4;
  }

  return out;
}

function encode(file, result) {
  const out = fs.createWriteStream(file);
  const step = Buffer.alloc(4);

  step.writeInt32LE(result.step, 0);
  out.write(step);

  out.write(encodeList(result.train.positives));
  out.write(encodeList(result.train.negatives));
  out.write(encodeList(result.validate.positives));
  out.write(encodeList(result.validate.negatives));

  out.end();
}

encode(OUT_DISTANCE, {
  step: DATA.step,
  train: crossDistance(DATA.train),
  validate: crossDistance(DATA.validate),
});
