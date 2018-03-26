#!/usr/bin/env npx ts-node

import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import * as propel from 'propel';

import { SHAPE } from '../src/dataset';

const FEATURE_COUNT = 18;
const MARGIN = propel.float32(0.1);
const ONE = propel.float32(1);
const EPSILON = propel.float32(0.000001);

const DATASETS_DIR = path.join(__dirname, '..', 'datasets');
const OUT_DIR = path.join(__dirname, '..', 'out');

const LABELS = require(path.join(DATASETS_DIR, 'index.json'));

interface IBulk {
  readonly input: Tensor;
  readonly labels: Tensor;
}

interface IBulkPair {
  readonly left: Tensor;
  readonly right: Tensor;
  readonly output: Tensor;
}

interface IParseCSVOptions {
  bulkSize: number;
  byLabel?: boolean;
}

function parseCSV(name, options: IParseCSVOptions = {}): ReadonlyArray<IBulk> {
  const file = path.join(OUT_DIR, name + '.csv');
  const content = fs.readFileSync(file);

  const labels: number[] = [];
  const tensors: propel.Tensor[] = [];

  // I know it looks lame, but we have not enough JS heap to parse it otherwise
  let last = 0;
  for (let off = 0; off < content.length; off++) {
    // '\n'
    if (content[off] !== 0x0a) {
      continue;
    }

    const line = content.slice(last, off).toString();
    last = off + 1;

    if (!line) {
      continue;
    }

    const parts = line.trim().split(/\s*,\s*/g);
    const label = parseInt(parts[0], 10);
    const input = new Float32Array(parts.length - 1);
    for (let i = 1; i < parts.length; i++) {
      input[i] = parseFloat(parts[i]);
    }

    const tensor = propel.tensor(input);

    labels.push(label);
    tensors.push(tensor);
  }

  const bulkSize = options.bulkSize;

  const bulks: IBulk[] = [];
  let indices: number = [];
  let lastLabel: number | undefined;

  // NOTE: assumes sorted validation dataset
  for (let i = 0; i < tensors.length; i++) {
    if (labels[i] !== lastLabel) {
      lastLabel = labels[i];
      indices.push(i);
    }
  }

  indices.push(tensors.length);
  for (let i = 0; i < indices.length - 1; i++) {
    const from = indices[i];
    const to = indices[i + 1];

    assert(labels.slice(from, to).every((elem) => elem === i));

    for (let j = from; j < to; j += bulkSize) {
      const input = tensors.slice(j, j + bulkSize);

      // Make sure that all bulks have the same size
      if (input.length !== bulkSize) {
        break;
      }

      bulks.push({
        input: propel.stack(input, 0),
        labels: propel.int32(labels.slice(j, j + bulkSize)),
      });
    }
  }
  return bulks;
}

console.time('parse');

const validateBulks = parseCSV('validate', { bulkSize: 64, byLabel: true });
const trainBulks = parseCSV('train', { bulkSize: 64, byLabel: true });

console.timeEnd('parse');

console.log('Loaded data, total labels %d', LABELS.length);
console.log('Train bulks: %d', trainBulks.length);
console.log('Validation bulks: %d', validateBulks.length);

function *bulkPairs(list: ReadonlyArray<IBulk>): Iterator<IBulkPair> {
  let left = list.length * list.length;
  while (left-- >= 0) {
    let i = Math.floor(Math.random() * list.length);
    let j = 0;
    do {
      j = Math.floor(Math.random() * list.length);
    } while (i === j);

    const left = list[i];
    const right = list[j];

    const output = left.labels.equal(right.labels).cast('float32');
    yield { left: left.input, right: right.input, output };
  }
}

function applySingle(input: Tensor, params: propel.Params): Tensor {
  const raw = input
    .linear("Features", params, FEATURE_COUNT).relu();

  return raw;
}

function apply(pair: IBulkPair, params: propel.Params): Tensor {
  const left = applySingle(pair.left, params);
  const right = applySingle(pair.right, params);

  // Euclidian distance^2
  return left.sub(right).square().reduceSum([ 1 ]).add(EPSILON).sqrt();
}

function contrastiveLoss(distance: Tensor, labels: Tensor): Tensor {
  return (
    ONE.sub(labels).mul(distance.square())
      .add(labels.mul(MARGIN.sub(distance).relu().square()))
  ).reduceMean();
}

async function validate(exp: propel.Experiment) {
  const params = exp.params;

  console.log('Validation:');

  let sum = 0;
  let count = 0;

  for (const pair of bulkPairs(validateBulks)) {
    const distance = apply(pair, params);
    const loss = contrastiveLoss(distance, pair.output);

    sum += loss.reduceMean().dataSync()[0];
    count++;
  }

  sum /= count;
  console.log('');
  console.log('  Mean loss %s', sum.toFixed(5));
  console.log('');
  for (const [ name, tensor ] of params) {
    if (/\/weights$/.test(name)) {
      const mean = tensor.square().reduceMean().dataSync()[0];
      console.log('  %s - mean=%s', name, mean.toFixed(5));
    }
  }
}

async function train(maxSteps?: number) {
  const exp = await propel.experiment("gradtype", { saveSecs: 10 });
  await validate(exp);

  let last: number | undefined;
  for (let repeat = 0; repeat < Infinity; repeat++) {
    for (const pair of bulkPairs(trainBulks)) {
      await exp.sgd({ lr: 0.01 }, (params) => {
        const distance = apply(pair, params)
        const loss = contrastiveLoss(distance, pair.output);
        return loss;

        let l2 = loss.zerosLike();
        for (const [ name, tensor ] of params) {
          if (/\/weights$/.test(name)) {
            l2 = l2.add(tensor.square().reduceMean());
          }
        }

        return loss.add(l2.mul(0.01));
      });

      if (maxSteps && exp.step >= maxSteps) return;
      if (last === undefined) {
        last = exp.step;
      } else if (exp.step - last > 5000) {
        await validate(exp);
        last = exp.step;
      }
    }
  }
}

train().catch((e) => {
  console.log(e);
});
