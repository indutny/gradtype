#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as propel from 'propel';

import { Collector, Dataset } from '../src/train-data';

const collector = new Collector();
const ds = new Dataset();

function print(exp: propel.Experiment): void {
  const input = ds.generateSingle(collector.getResult());
  const params = exp.params;

  const result = propel.tensor(input).expandDims(0)
    .linear("L5", params, 200).relu()
    .linear("Adjust", params, 3)
    .softmax();

  const max = result.argmax(1).dataSync()[0];
  console.log('');
  console.log(max === 0 ? 'both' : max === 1 ? 'left' : 'right');
  console.log(Array.from(result.dataSync()).map(
    (val) => (val * 100).toFixed(2) + '%'));
}

async function run() {
  const exp = await propel.experiment("gradtype");

  let counter = 0;
  process.stdin.on('data', (input) => {
    for (let i = 0; i < input.length; i++) {
      const code = input[i];
      if (code === 3 || code === 4) {
        process.exit(0);
        return;
      }

      collector.register(code);
      if (++counter % 10 === 0) {
        print(exp);
      }

      // backspace
      if (code === 127) {
        input[i] = 8;
      }
    }
    process.stdout.write(input);
  });
  process.stdin.setRawMode(true);

  console.log('ready');
}

run();
