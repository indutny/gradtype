#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as path from 'path';

const a = JSON.parse(fs.readFileSync(process.argv[2]).toString());
const b = JSON.parse(fs.readFileSync(process.argv[3]).toString());

process.stdout.write(JSON.stringify(a.concat(b)));
