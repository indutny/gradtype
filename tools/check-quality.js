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

function computeStats(subdata, priorSame, truePositives, falseNegatives) {
  const stats = {
    truePositives,
    trueNegatives: subdata.negatives.length - falseNegatives + 1,

    lessGivenSame: truePositives / subdata.positives.length,
    lessGivenDiff: falseNegatives / subdata.negatives.length,

    // Filled below
    accuracy: 0,
    sameGivenLess: 0,
  };

  const priorDiff = 1 - priorSame;

  stats.accuracy = (stats.truePositives + stats.trueNegatives) /
      (subdata.positives.length + subdata.negatives.length);
  stats.sameGivenLess = stats.lessGivenSame * priorSame /
    (stats.lessGivenSame * priorSame + stats.lessGivenDiff * priorDiff);

  return stats;
}

function optimize(subdata, priorSame, confidence) {
  let result = null;
  let best = 0;

  let j = 0;
  for (let i = 0; i < subdata.positives.length; i++) {
    const cutoff = subdata.positives[i];
    while (subdata.negatives[j] < cutoff) {
      j++;
    }

    const stats = computeStats(subdata, priorSame, i + 1, j + 1);

    const size = computeSize(stats, confidence);
    if (stats.accuracy > best) {
      best = stats.accuracy;
      const min = size * stats.lessGivenSame;
      result = { size, cutoff, stats, min };
    }
  }
  return result;
}

function check(subdata, priorSame, confidence, cutoff) {
  let i = bounds.ge(subdata.positives, cutoff);
  let j = bounds.ge(subdata.negatives, cutoff);

  const stats = computeStats(subdata, priorSame, i + 1, j + 1);

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
