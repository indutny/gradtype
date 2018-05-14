#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as path from 'path';

const json = JSON.parse(fs.readFileSync(process.argv[2]).toString());

const keys = json.map((e) => e.k === 'Backspace' ? '<' : e.k).join('');

console.log(keys);
