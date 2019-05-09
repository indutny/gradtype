'use strict';

const weights = require('./weights.json');
const features = require('./features.json');
const Model = require('../src/model');

const m = new Model(weights);

function seq(entry, early = true) {
  const seq = [];

  for (let i = 0; i < entry.codes.length; i++) {
    const code = entry.codes[i];
    const duration = entry.deltas[i];
    const hold = entry.holds[i];

    if (early && code === 0) {
      break;
    }

    seq.push({ code: code - 1, hold, duration });
  }
  return seq;
}

const vec1 = m.call(seq(features.train[0]));
console.log(vec1, features.train[0].features);
