'use strict';

const weights = require('./weights.json');
const features = require('./features.json');
const Model = require('../src/model');

const m = new Model(weights);

const seq = [];

const entry = features.train[0];
for (let i = 0; ; i++) {
  const code = entry.codes[i];
  const duration = entry.deltas[i];
  const hold = entry.holds[i];

  if (code === 0) {
    break;
  }

  seq.push({ code: code - 1, hold, duration });
}

const vec1 = m.call(seq);
console.log(vec1, entry.features);
