#!/usr/bin/env npx ts-node

import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import * as propel from 'propel';

import { SHAPE } from '../src/dataset';

const OUT_DIR = path.join(__dirname, '..', 'out');

const LABELS = require(path.join(OUT_DIR, 'labels.json'));

interface IBulk {
  readonly input: Tensor;
  readonly labels: Tensor;
}

interface INNOutput {
  readonly output: Tensor;
  readonly l1: Tensor;
}

function parseCSV(name, bulkSize?: number): ReadonlyArray<IBulk> {
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

  if (bulkSize === undefined) {
    bulkSize = tensors.length;
  }

  const bulks: IBulk[] = [];
  for (let i = 0; i < tensors.length; i += bulkSize) {
    bulks.push({
      input: propel.stack(tensors.slice(i, i + bulkSize), 0),
      labels: propel.tensor(labels.slice(i, i + bulkSize), {
        dtype: 'int32'
      }).oneHot(LABELS.length),
    });
  }

  return bulks;
}

const trainBulks = parseCSV('train', 100);
const validateBulks = parseCSV('validate');

console.log('Loaded data, total labels %d', LABELS.length);

function apply(bulk: IBulk, params: propel.Params): INNOutput {
  const l1 = bulk.input
    .linear("L1", params, 200);

  const output = l1
    .linear("Adjust", params, LABELS.length);

  return {
    output,
    l1
  };
}

async function validate(exp: propel.Experiment) {
  const params = exp.params;

  let sum = 0;
  let count = 0;
  const l1List = [];
  for (const bulk of validateBulks) {
    const { l1, output } = apply(bulk, params);

    const loss = output
      .argmax(1)
      .equal(bulk.labels.argmax(1).cast('int32'))
      .reduceMean();

    sum += loss.dataSync()[0];
    count++;
    l1List.push(l1);
  }
  const l1 = l1List.length === 1 ? l1List[0] : propel.stack(l1List, 0);

  console.log('Success rate: %s %%', (100 * sum / count).toFixed(2));
  console.log('L1 moments:', l1.moments());
}

async function train(maxSteps?: number) {
  const exp = await propel.experiment("gradtype", { saveSecs: 10 });
  await validate(exp);

  let last: number | undefined;
  for (let repeat = 0; repeat < Infinity; repeat++) {
    for (const bulk of trainBulks) {
      await exp.sgd({ lr: 0.01 }, (params) =>
        apply(bulk, params)
          .output
          .softmaxLoss(bulk.labels));

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
