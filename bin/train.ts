#!/usr/bin/env npx ts-node

import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import * as propel from 'propel';

import { SHAPE } from '../src/dataset';
import { shuffle } from '../src/utils';

import Tensor = propel.Tensor;

const FEATURE_COUNT = 18;
const BATCH_SIZE = 32;

const DATASETS_DIR = path.join(__dirname, '..', 'datasets');
const OUT_DIR = path.join(__dirname, '..', 'out');

const LABELS = require(path.join(DATASETS_DIR, 'index.json'));

interface IBatchElem {
  readonly input: Tensor;
  readonly label: Tensor;
}

interface IBatch {
  readonly anchor: IBatchElem;
  readonly positive: IBatchElem;
  readonly negative: IBatchElem;
}

interface IParseCSVOptions {
  readonly batchSize: number;
}

interface IApplyRenegative {
  readonly positive: Tensor;
  readonly negative: Tensor;
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
  for (let i = 0; i < tensors.length; i += batchSize * 3) {
    const avail = Math.min(batchSize * 3, tensors.length - i) / 3;

    // Can't do `stack` on less
    if (avail < 2) {
      break;
    }

    const anchor: Tensor[] = [];
    const anchorLabels: number[] = [];
    const positive: Tensor[] = [];
    const positiveLabels: number[] = [];
    const negative: Tensor[] = [];
    const negativeLabels: number[] = [];
    for (let j = 0; j < avail; j++) {
      const anchorIndex = i + j * 3;
      const positiveIndex = anchorIndex + 1;
      const negativeIndex = anchorIndex + 2;

      anchor.push(tensors[anchorIndex]);
      positive.push(tensors[positiveIndex]);
      negative.push(tensors[negativeIndex]);
      anchorLabels.push(labels[anchorIndex]);
      positiveLabels.push(labels[positiveIndex]);
      negativeLabels.push(labels[negativeIndex]);
    }

    const anchorLabelsT = propel.int32(anchorLabels);
    const positiveLabelsT = propel.int32(positiveLabels);
    const negativeLabelsT = propel.int32(negativeLabels);

    batches.push({
      anchor: { input: propel.stack(anchor, 0), label: anchorLabelsT },
      positive: { input: propel.stack(positive, 0), label: positiveLabelsT },
      negative: { input: propel.stack(negative, 0), label: negativeLabelsT },
    });
  }
  return batches;
}

console.time('parse');

const validateBatches = parseCSV('validate', { batchSize: BATCH_SIZE });
const trainBatches = parseCSV('train', { batchSize: BATCH_SIZE });

console.timeEnd('parse');

console.log('Loaded data, total labels %d', LABELS.length);
console.log('Train batches: %d', trainBatches.length);
console.log('Validation batches: %d', validateBatches.length);

function applySingle(input: Tensor, params: propel.Params): Tensor {
  return input
    .linear("Features", params, FEATURE_COUNT);
}

function distanceSquare(a: Tensor, b: Tensor): Tensor {
  return a.sub(b).square();
}

function apply(batch: IBatch, params: propel.Params): IApplyResult {
  const anchor = applySingle(batch.anchor.input, params);
  const positive = applySingle(batch.positive.input, params);
  const negative = applySingle(batch.negative.input, params);

  return {
    positive: distanceSquare(anchor, positive),
    negative: distanceSquare(anchor, negative),
  };
}

function computeLoss(output: IApplyResult): Tensor {
  // Triplet loss
  return output.positive.reduceMean()
    .sub(output.negative.reduceMean()).add(0.2)
    .reduceMean().relu();
}

async function validate(exp: propel.Experiment, batches: IBatch[]) {
  const params = exp.params;

  let sum = 0;
  let count = 0;
  let meanPositive = 0;
  let meanNegative = 0;

  for (const batch of batches.slice(0, validateBatches.length)) {
    const output = apply(batch, params);

    meanPositive += output.positive.sqrt().reduceMean().dataSync()[0];
    meanNegative += output.negative.sqrt().reduceMean().dataSync()[0];

    const positive = output.positive.sqrt().less(0.5).cast('int32')
      .reduceMean().dataSync()[0];
    const negative = output.negative.sqrt().greater(0.5).cast('int32')
      .reduceMean().dataSync()[0];

    sum += positive + negative;
    count++;
  }

  sum /= count * 2;
  meanPositive /= count;
  meanNegative /= count;

  console.log('');
  console.log('  Success rate %s %%', (sum * 100).toFixed(3));
  console.log('  Mean positive distance %s', meanPositive.toFixed(5));
  console.log('  Mean negative distance %s', meanNegative.toFixed(5));
  console.log('');
}

async function train(maxSteps?: number) {
  const exp = await propel.experiment("gradtype", { saveSecs: 10 });
  await validate(exp, validateBatches);

  let last: number | undefined;
  for (let repeat = 0; repeat < Infinity; repeat++) {
    shuffle(trainBatches);
    for (const batch of trainBatches) {
      await exp.sgd({ lr: 0.01 }, (params) => {
        const output = apply(batch, params)
        const loss = computeLoss(output);

        let l2 = loss.zerosLike();
        for (const [ name, tensor ] of params) {
          if (/\/weights$/.test(name)) {
            l2 = l2.add(tensor.square().reduceMean());
          }
        }

        return loss.add(l2.mul(0.01));
      });

      if (maxSteps && exp.step >= maxSteps) return;

      // Validate every 5000 steps
      if (last === undefined) {
        last = exp.step;
      } else if (exp.step - last > 5000) {
        console.log('Training dataset:');
        await validate(exp, trainBatches);

        console.log('Validation dataset:');
        await validate(exp, validateBatches);
        last = exp.step;
      }
    }
  }
}

train().catch((e) => {
  console.log(e);
});
