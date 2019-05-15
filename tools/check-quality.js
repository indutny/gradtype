'use strict';

const fs = require('fs');

const DATA = JSON.parse(fs.readFileSync(process.argv[2]).toString());

const PRIOR = 0.05;
const TARGET = 0.999;

// Totally arbitrary, depends on PRIOR
const TWEAK = Math.exp(1.2);

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

function score(pos, distances) {
  let negMin = Infinity;
  let negHit = 0;
  for (const d of distances.negatives) {
    negHit += d > pos ? 1 : 0;
    negMin = Math.min(negMin, d);
  }

  let posMax = 0;
  let posHit = 0;
  for (const d of distances.positives) {
    posHit += d < pos ? 1 : 0;
    posMax = Math.max(posMax, d);
  }

  const negCount = distances.negatives.length;
  const posCount = distances.positives.length;
  const total = negCount + posCount;

  return {
    lessGivenSame: posHit / posCount,
    greaterGivenDiff: negHit / negCount,
    negMin,
    posMax,
  };
}

function isSame(pos, known, unknown) {
  const distances = [];
  for (const features of known) {
    distances.push(distance(features, unknown));
  }

  distances.sort();
  let mean = 0;
  let count = Math.min(20, distances.length);
  for (let i = 0; i < count; i++) {
    mean += distances[i];
  }
  mean /= count;

  return mean < pos;
}

function scoreByCat(pos, map) {
  let diffHit = 0;
  let diffTotal = 0;

  let sameHit = 0;
  let sameTotal = 0;
  let perKey = 0;
  let perKeyCount = 0;
  let keyMap = new Map();
  for (const [ key, list ] of map.entries()) {
    let keyTotal = 0;
    let keySame = 0;
    let keyDiff = 0;

    for (const [ otherKey, otherList ] of map.entries()) {
      if (otherKey === key) {
        continue;
      }

      let catKeyDiff = 0;
      for (const otherFeatures of otherList) {
        catKeyDiff += isSame(pos, list, otherFeatures) ? 0 : 1;
      }
      catKeyDiff /= otherList.length;

      keyDiff += catKeyDiff;
      keyTotal++;
      diffTotal++;
    }

    for (let i = 0; i < list.length; i++) {
      sameTotal++;
      keyTotal++;
      keySame += isSame(pos, list.slice(0, i).concat(list.slice(i + 1)),
        list[i]);
    }

    sameHit += keySame;
    diffHit += keyDiff;

    const keyMean = (keySame + keyDiff) / keyTotal;
    perKey += keyMean;
    perKeyCount++;

    keyMap.set(key, keyMean);
  }
  perKey /= perKeyCount;

  return { diff: diffHit / diffTotal, same: sameHit / sameTotal, perKey };
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

const featuresByCategory = {
  train: byCategory(DATA.train),
  validate: byCategory(DATA.validate),
};

const trainPos = 9 || search(distances.train);
const trainScore = score(trainPos, distances.train);
const valScore = score(trainPos, distances.validate);

console.log('train pos=%d', trainPos);
console.log('train score=%j', trainScore);
console.log('val score=%j', valScore);

// Bayes in all its beauty
function sameGivenLess(score, sameProb) {
  const less = score.lessGivenSame * sameProb +
    (1 - score.greaterGivenDiff) * (1 - sameProb);
  return score.lessGivenSame * sameProb / less;
}

function trials(target, prob) {
  return Math.log(1 - target) / Math.log(1 - prob);
}

const trainSL = sameGivenLess(trainScore, PRIOR);
const valSL = sameGivenLess(valScore, PRIOR);

console.log('prior=%d target=%d', PRIOR, TARGET);
console.log('train same given less', trainSL);
console.log('train trials', trials(TARGET, trainSL));
console.log('validate same given less', valSL);
console.log('val trials', trials(TARGET, valSL));

const trainCatScore = scoreByCat(trainPos, featuresByCategory.train);
const valCatScore = scoreByCat(trainPos, featuresByCategory.validate);

console.log('train score by cat', trainCatScore);
console.log('train multiply',
  trials(TARGET, Math.min(trainCatScore.same, trainCatScore.diff)));
console.log('val score by cat', valCatScore);
console.log('val multiply',
  trials(TARGET, Math.min(valCatScore.same, valCatScore.diff)));
