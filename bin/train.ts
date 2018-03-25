#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as propel from 'propel';

import { TrainData } from '../src/gradtype';

const td = new TrainData();

const { maxIndex, bulks } =
  td.parse(fs.readFileSync(process.argv[2]).toString());

async function train(maxSteps?: number) {
  const exp = await propel.experiment("gradtype");

  for (let repeat = 0; repeat < 1000; repeat++) {
    for (const bulk of bulks) {
      await exp.sgd({ lr: 0.01 }, (params) =>
        bulk.input
          .linear("L1", params, 64).relu()
          .linear("L2", params, 32).relu()
          .linear("L3", params, maxIndex + 1)
          .softmaxLoss(bulk.labels));

      if (maxSteps && exp.step >= maxSteps) return;
    }
  }
}

train();
