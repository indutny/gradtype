'use strict';

const fs = require('fs');

const DATA = JSON.parse(fs.readFileSync(process.argv[2]).toString());

const PRIOR = 0.05;
const TARGET = 0.999;

// Totally arbitrary, depends on PRIOR
const TWEAK = Math.exp(1);

function distance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] - b[i]) ** 2;
  }
  return Math.sqrt(sum);
}

function crossDistance(data) {
  const positives = [];
  const negatives = [];

  for (let i = 0; i < data.length; i++) {
    const category = data[i].category;
    const features = data[i].features;

    for (let j = i + 1; j < data.length; j++) {
      const otherCategory = data[j].category;
      const otherFeatures = data[j].features;

      const d = distance(features, otherFeatures);
      if (category === otherCategory) {
        positives.push(d);
      } else {
        negatives.push(d);
      }
    }
  }

  return { positives, negatives };
}

function score(pos, distances) {
  let negHit = 0;
  for (const d of distances.negatives) {
    negHit += d > pos ? 1 : 0;
  }
  let posHit = 0;
  for (const d of distances.positives) {
    posHit += d < pos ? 1 : 0;
  }

  const negCount = distances.negatives.length;
  const posCount = distances.positives.length;
  const total = negCount + posCount;

  return {
    lessGivenSame: posHit / posCount,
    greaterGivenDiff: negHit / negCount,
  };
}

function generalScore(distances) {
  const negCount = distances.negatives.length;
  const posCount = distances.positives.length;
  const total = negCount + posCount;
  return {
    diff: negCount / total,
    same: posCount / total,
  };
}

function loss(s) {
  return -(TWEAK * s.greaterGivenDiff + s.lessGivenSame);
}

function search(distances) {
  let low = Infinity;
  let high = 0;

  for (const d of distances.negatives) {
    low = Math.min(low, d);
  }
  for (const d of distances.positives) {
    high = Math.max(high, d);
  }

  console.error('Search between: %d and %d', low, high);

  while (high - low > 1e-3) {
    const ls = loss(score(low, distances));
    const hs = loss(score(high, distances));

    const mid = (high + low) / 2;

    if (ls < hs) {
      high = mid;
    } else {
      low = mid;
    }
  }
  return high;
}

const distances = {
  train: crossDistance(DATA.train),
  validate: crossDistance(DATA.validate),
};

const trainPos = search(distances.train);
const trainScore = score(trainPos, distances.train);
const valScore = score(trainPos, distances.validate);

console.log('train pos=%d', trainPos);
console.log('train score=%j', trainScore);
console.log('val score=%j', valScore);

// See: http://mathb.in/33460
function sameGivenLess(score, sameProb) {
  const less = score.lessGivenSame * sameProb +
    (1 - score.greaterGivenDiff) * (1 - sameProb);
  return score.lessGivenSame * sameProb / less;
}

const trainSL = sameGivenLess(trainScore, PRIOR);
const valSL = sameGivenLess(valScore, PRIOR);

console.log('prior=%d target=%d', PRIOR, TARGET);
console.log('train same given less', trainSL);
console.log('train trials', Math.log(1 - TARGET) / Math.log(1 - trainSL));
console.log('validate same given less', valSL);
console.log('val trials', Math.log(1 - TARGET) / Math.log(1 - valSL));
