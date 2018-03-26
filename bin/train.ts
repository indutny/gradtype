#!/usr/bin/env npx ts-node

import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import * as propel from 'propel';

import { SHAPE } from '../src/dataset';

import Tensor = propel.Tensor;

const FEATURE_COUNT = 18;
const MARGIN = propel.float32(0.1);
const ONE = propel.float32(1);
const EPSILON = propel.float32(0.000001);

const DATASETS_DIR = path.join(__dirname, '..', 'datasets');
const OUT_DIR = path.join(__dirname, '..', 'out');

const LABELS = require(path.join(DATASETS_DIR, 'index.json'));

interface IBatch {
  readonly left: Tensor;
  readonly leftLabels: Tensor;
  readonly right: Tensor;
  readonly rightLabels: Tensor;
  readonly output: Tensor;
}

interface IParseCSVOptions {
  batchSize: number;
}

function parseCSV(name: string, options: IParseCSVOptions): ReadonlyArray<IBatch> {
  const file = path.join(OUT_DIR, name + '.csv');
  const content = fs.readFileSync(file);

  const labels: number[] = [];
  const tensors: Tensor[] = [];

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

  const batchSize = options.batchSize;

  const batches: IBatch[] = [];
  for (let i = 0; i < tensors.length; i += batchSize * 2) {
    const avail = Math.min(batchSize * 2, tensors.length - i) / 2;

    const left: Tensor[] = [];
    const leftLabels: number[] = [];
    const right: Tensor[] = [];
    const rightLabels: number[] = [];

    for (let j = 0; j < avail; j++) {
      left.push(tensors[i + j * 2]);
      leftLabels.push(labels[i + j * 2]);
      right.push(tensors[i + j * 2 + 1]);
      rightLabels.push(labels[i + j * 2 + 1]);
    }

    const leftLabelsT = propel.int32(leftLabels);
    const rightLabelsT = propel.int32(rightLabels);

    batches.push({
      left: propel.stack(left, 0),
      leftLabels: leftLabelsT,
      right: propel.stack(right, 0),
      rightLabels: rightLabelsT,
      output: leftLabelsT.equal(rightLabelsT).cast('float32'),
    });
  }
  return batches;
}

console.time('parse');

const validateBatches = parseCSV('validate', { batchSize: 64 });
const trainBatches = parseCSV('train', { batchSize: 64 });

console.timeEnd('parse');

console.log('Loaded data, total labels %d', LABELS.length);
console.log('Train batches: %d', trainBatches.length);
console.log('Validation batches: %d', validateBatches.length);

function applySingle(input: Tensor, params: propel.Params): Tensor {
  const raw = input
    .linear("Features", params, FEATURE_COUNT).relu();

  return raw.logSoftmax();
}

function apply(batch: IBatch, params: propel.Params): Tensor {
  const left = applySingle(batch.left, params);
  const right = applySingle(batch.right, params);

  // Euclidian distance^2
  return left.sub(right).square().reduceSum([ 1 ]).add(EPSILON).sqrt();
}

function contrastiveLoss(distance: Tensor, output: Tensor): Tensor {
  return (
    ONE.sub(output).mul(distance.square())
      .add(output.mul(MARGIN.sub(distance).relu().square()))
  ).reduceMean();
}

async function validate(exp: propel.Experiment) {
  const params = exp.params;

  console.log('Validation:');

  let sum = 0;
  let count = 0;

  for (const batch of validateBatches) {
    const distance = apply(batch, params);
    const loss = contrastiveLoss(distance, batch.output)
      .reduceMean().dataSync()[0];

    sum += loss;
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
    for (const batch of trainBatches) {
      await exp.sgd({ lr: 0.01 }, (params) => {
        const distance = apply(batch, params)
        const loss = contrastiveLoss(distance, batch.output);
        return loss;
      });

      if (maxSteps && exp.step >= maxSteps) return;

      // Validate every 5000 steps
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
