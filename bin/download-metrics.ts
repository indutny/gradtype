#!/usr/bin/env npx ts-node

import fetch from 'node-fetch';

const HOST = process.env.GRADTYPE_HOST || '127.0.0.1';
const RUN = process.env.GRADTYPE_RUN || 'ds396-cleanup-lr_0.001';
const PORT = 6006;

const BASE_URI =
  `http://${HOST}:${PORT}/data/plugin/scalars/scalars?run=${RUN}&format=json`;

async function downloadSingle(tag: string) {
  const uri = `${BASE_URI}&tag=${encodeURIComponent(tag)}`;

  const res = await fetch(uri);
  return await res.json();
}

async function download() {
  const metrics = [
    'negative_5',
    'negative_10',
    'negative_25',
    'negative_50',
    'negative_75',
    'negative_90',
    'negative_95',
    'positive_5',
    'positive_10',
    'positive_25',
    'positive_50',
    'positive_75',
    'positive_90',
    'positive_95',
  ];

  const tags = [];
  for (const category of [ 'train', 'validate' ]) {
    for (const metric of metrics) {
      tags.push(`${category}/${metric}`);
    }
  }

  const data = await Promise.all(tags.map(async (tag) => {
    return {
      tag,
      data: await downloadSingle(tag),
    };
  }));

  let minSteps = Infinity;
  for (const tag of data) {
    minSteps = Math.min(minSteps, tag.data.length);
  }

  const labels: string[] = [ 'Wall time', 'Step' ].concat(tags);

  const rows: Array<number[]> = [];
  for (let i = 0; i < minSteps; i++) {
    const newRow: number[] = data[0].data[i].slice(0, 2);
    for (const tag of data) {
      const sparse = Math.round(i / minSteps * tag.data.length);
      newRow.push(tag.data[sparse][2]);
    }
    rows.push(newRow);
  }

  const csv = labels.join(',') + '\n' +
    rows.map((row) => row.join(',')).join('\n');
  return csv;
}

download().then((csv) => {
  process.stdout.write(csv);
}).catch((e) => {
  console.error(e);
  process.exit(1);
});
