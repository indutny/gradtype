'use strict';

const fs = require('fs');
const bounds = require('binary-search-bounds');

function loadList(buf, off, out) {
  const len = buf.readInt32LE(off);
  off += 4;

  for (let i = 0; i < len; i++) {
    const val = buf.readFloatLE(off);
    off += 4;

    out.push(val);
  }

  return off;
}

function load(fname) {
  const buf = fs.readFileSync(fname);
  let off = 0;

  const steps = buf.readInt32LE(off);
  off += 4;

  const result = {
    steps,
    train: { positives: [], negatives: [] },
    validate: { positives: [], negatives: [] },
  };

  off = loadList(buf, off, result.train.positives);
  off = loadList(buf, off, result.train.negatives);
  off = loadList(buf, off, result.validate.positives);
  off = loadList(buf, off, result.validate.negatives);
  return result;
}

const DATA = load(process.argv[2]);
const PRIOR_SAME = 1 / require('../datasets').length;
const CONFIDENCE = 0.99;

// 1 - Confidence = (1 - sameGivenLess) ** (categoryLen * lessGivenSame)
// categoryLen = (Math.log(1 - Confidence) / Math.log(1 - sameGivenLess)) /
//   lessGivenSame

function computeSize(stats, confidence) {
  let size = Math.log(1 - confidence) / Math.log(1 - stats.sameGivenLess);
  size /= stats.lessGivenSame;
  return size;
}

function optimize(subdata, priorSame, confidence) {
  let result = null;
  let best = Infinity;

  const priorDiff = 1 - priorSame;
  let j = 0;
  for (let i = 0; i < subdata.positives.length; i++) {
    const cutoff = subdata.positives[i];
    while (subdata.negatives[j] < cutoff) {
      j++;
    }

    const stats = {
      lessGivenSame: (i + 1) / subdata.positives.length,
      lessGivenDiff: (j + 1) / subdata.negatives.length,
      sameGivenLess: 0,
    };

    stats.sameGivenLess = stats.lessGivenSame * priorSame /
      (stats.lessGivenSame * priorSame + stats.lessGivenDiff * priorDiff);

    const size = computeSize(stats, confidence);
    if (size < best) {
      best = size;
      const min = size * stats.lessGivenSame;
      result = { size, cutoff, stats, min };
    }
  }
  return result;
}

function check(subdata, priorSame, confidence, cutoff) {
  let i = bounds.ge(subdata.positives, cutoff);
  let j = bounds.ge(subdata.negatives, cutoff);

  const priorDiff = 1 - priorSame;
  const stats = {
    lessGivenSame: (i + 1) / subdata.positives.length,
    lessGivenDiff: (j + 1) / subdata.negatives.length,
    sameGivenLess: 0,
  };

  stats.sameGivenLess = stats.lessGivenSame * priorSame /
    (stats.lessGivenSame * priorSame + stats.lessGivenDiff * priorDiff);

  const size = computeSize(stats, confidence);
  const min = size * stats.lessGivenSame;
  return { size, cutoff, stats, min };
}

console.log('target confidence', CONFIDENCE);
console.log('prior same', PRIOR_SAME);

const trainResult = optimize(DATA.train, PRIOR_SAME, CONFIDENCE);
console.log('train', trainResult);

const validateResult = check(DATA.validate, PRIOR_SAME, CONFIDENCE,
  trainResult.cutoff);
console.log('validate', validateResult);
