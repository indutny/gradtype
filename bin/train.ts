#!/usr/bin/env npx ts-node

import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import * as propel from 'propel';

import { SHAPE } from '../src/dataset';

const DATASETS_DIR = path.join(__dirname, '..', 'datasets');
const OUT_DIR = path.join(__dirname, '..', 'out');

const LABELS = require(path.join(DATASETS_DIR, 'index.json'));

interface IBulk {
  readonly input: Tensor;
  readonly labels: Tensor;
}

interface INNOutput {
  readonly output: Tensor;
  readonly l1: Tensor;
}

interface IParseCSVOptions {
  bulkSize?: number;
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

  const bulks: IBulk[] = [];
  if (options.byLabel === true) {
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

      bulks.push({
        input: propel.stack(tensors.slice(from, to), 0),
        labels: propel.tensor(labels.slice(from, to), {
          dtype: 'int32'
        }).oneHot(LABELS.length),
      });
    }
    return bulks;
  }

  let bulkSize: number = options.bulkSize === undefined ?
    tensors.length : options.bulkSize;

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

const validateBulks = parseCSV('validate', { byLabel: true });
const trainBulks = parseCSV('train', { bulkSize: 100 });

console.log('Loaded data, total labels %d', LABELS.length);
console.log('Train bulks: %d', trainBulks.length);
console.log('Validation bulks: %d', validateBulks.length);

function apply(bulk: IBulk, params: propel.Params): INNOutput {
  const l1 = bulk.input
    .linear("L1", params, 200).relu();

  const output = l1
    .linear("Adjust", params, LABELS.length);

  return {
    output,
    l1
  };
}

async function validate(exp: propel.Experiment) {
  const params = exp.params;

  console.log('Validation:');

  assert.strictEqual(validateBulks.length, LABELS.length);
  for (let i = 0; i < validateBulks.length; i++) {
    const bulk = validateBulks[i];
    const { l1, output } = apply(bulk, params);

    const success = output
      .argmax(1)
      .equal(bulk.labels.argmax(1).cast('int32'))
      .reduceMean()
      .dataSync()[0];

    const { mean, variance } = l1.moments();
    console.log('  %s - %s %%, activation: mean=%s variance=%s',
      LABELS[i], (100 * success).toFixed(2), mean.dataSync()[0].toFixed(4),
      variance.dataSync()[0].toFixed(4));
  }
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
