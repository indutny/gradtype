#!/usr/bin/env npx ts-node

import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import * as propel from 'propel';

const OUT_DIR = path.join(__dirname, '..', 'out');

const labels = require(path.join(OUT_DIR, 'labels.json'));

function parseCSV(name) {
  const file = path.join(OUT_DIR, name + '.csv');
  const content = fs.readFileSync(file).toString();

  const lines = content.split(/\n/g);
  for (const line of lines) {
    if (!line) {
      continue;
    }
    console.log(line);
  }
}

const train = parseCSV('train');
const validate = parseCSV('validate');

/*
function apply(bulk: ITrainDataBulk, params: propel.Params): propel.Tensor {
  return bulk.input
    .linear("Adjust", params, maxIndex + 1).relu();
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
      await exp.sgd({ lr: 0.03 }, (params) =>
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
 */
