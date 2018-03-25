#!/usr/bin/env npx ts-node

import * as assert from 'assert';
import * as fs from 'fs';
import * as propel from 'propel';

import { ITrainDataBulk, TrainData } from '../src/gradtype';

const td = new TrainData();

const { maxIndex, bulks } =
  td.parse(fs.readFileSync(process.argv[2]).toString());

const verifyData =
  td.parse(fs.readFileSync(process.argv[3]).toString());

assert.strictEqual(verifyData.maxIndex, maxIndex);

function apply(bulk: ITrainDataBulk, params: propel.Params): propel.Tensor {
  return bulk.input
    .linear("L2", params, 200).relu()
    .linear("L5", params, maxIndex + 1);
}

async function verify(exp: propel.Experiment) {
  const params = exp.params;

  let sum = 0;
  let count = 0;
  for (const bulk of verifyData.bulks) {
    const loss = apply(bulk, params)
      .argmax(1)
      .equal(bulk.labels.argmax(1).cast('int32'))
      .reduceMean();

    sum += loss.dataSync()[0];
    count++;
  }

  console.log('Success rate: %s %%', (100 * sum / count).toFixed(2));
}

async function train(maxSteps?: number) {
  const exp = await propel.experiment("gradtype", { saveSecs: 10 });
  await verify(exp);

  let last: number | undefined;
  for (let repeat = 0; repeat < 1000000; repeat++) {
    for (const bulk of bulks) {
      await exp.sgd({ lr: 0.01 }, (params) =>
        apply(bulk, params)
          .softmaxLoss(bulk.labels));

      if (maxSteps && exp.step >= maxSteps) return;
      if (last === undefined) {
        last = exp.step;
      } else if (exp.step - last > 5000) {
        await verify(exp);
        last = exp.step;
      }
    }
  }
}

train().catch((e) => {
  console.log(e);
});
