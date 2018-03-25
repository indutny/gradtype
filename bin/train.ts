#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as propel from 'propel';

const input = fs.readFileSync(process.argv[2]).toString()
  .split(/\r\n|\r|\n/g).map((line) => line.trim()).filter((line) => line);

const recordings: Tensor[] = [];
const indices: number[] = [];
let maxIndex: number = 0;
while (input.length !== 0) {
  // Consume less memory
  const line = input.pop();
  const parts = line.split(/\s*,\s*/g);

  if (input.length % 500 === 0) {
    console.log('left to load %d', input.length);
  }

  const single = new Float32Array(parts.length - 1);
  for (let i = 0; i < single.length; i++) {
    single[i] = parseFloat(parts[i]);
  }
  const index = parts[parts.length - 1];
  maxIndex = Math.max(maxIndex, index);

  indices.push(index);
  recordings.push(propel.tensor(single));
}

async function train(maxSteps) {
  const exp = await propel.experiment("gradtype");
  const BULK = 500;

  const dataset: { labels: propel.Tensor, input: propel.Tensor}[] = [];

  for (let i = 0; i < indices.length; i += BULK) {
    const labels = propel.tensor(indices.slice(i, i + BULK), { dtype: 'int32' })
      .oneHot(maxIndex + 1);
    const input = propel.stack(recordings.slice(i, i + BULK), 0);

    dataset.push({ labels, input });
  }

  for (let repeat = 0; repeat < 1000; repeat++) {
    for (const bulk of dataset) {
      await exp.sgd({ lr: 0.01 }, (params) =>
        bulk.input
          .linear("L1", params, 200).relu()
          .linear("L2", params, 100).relu()
          .linear("L3", params, maxIndex + 1)
          .softmaxLoss(bulk.labels));

      if (maxSteps && exp.step >= maxSteps) break;
    }
  }
}

train(3000);
