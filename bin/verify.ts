#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as propel from 'propel';

import { TrainData } from '../src/gradtype';

const td = new TrainData();

const { maxIndex, bulks } =
  td.parse(fs.readFileSync(process.argv[2]).toString());

async function verify() {
  const exp = await propel.experiment("gradtype");

  let sum = 0;
  let count = 0;
  for (const bulk of bulks) {
    const params = exp.params;

    const loss = bulk.input
      .linear("L2", params, 20).relu()
      .linear("L3", params, 40).relu()
      .linear("L5", params, maxIndex + 1)
      .argmax(1)
      .equal(bulk.labels.argmax(1).cast('int32'))
      .reduceMean();

    sum += loss.dataSync()[0];
    count++;
  }

  console.log('Success rate: %s %%', (100 * sum / count).toFixed(2));
}

verify();
