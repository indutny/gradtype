#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as path from 'path';
import { Collector } from '../src/collector';

const collector = new Collector();

const raw = fs.readFileSync(path.join(__dirname, '..', 'data', 'wilde.txt'));
const text: string = raw.toString().replace(/\s+/g, ' ');
const sentences = text.split(/\.+(?!")/g).map((line) => line.trim());

let index = Math.floor(Math.random() * sentences.length);

function terminate() {
  fs.writeFileSync('./gradtype.json', JSON.stringify(collector.getResult()));
  process.exit(0);
}

function display() {
  const sentence = sentences[index];
  index++;
  if (index >= sentences.length) {
    index = 0;
  }

  const onInput = (input) => {
    let dot = false;
    for (let i = 0; i < input.length; i++) {
      const code = input[i];
      if (code === 3 || code === 4) {
        process.stdin.removeListener('data', onInput);
        terminate();
        return;
      }

      collector.register(code);

      // '.'
      if (code === 46) {
        dot = true;
        process.stdin.removeListener('data', onInput);
        display();
        break;
      }

      // backspace
      if (code === 127) {
        input[i] = 8;
      }
    }
    process.stdout.write(dot ? input.slice(0, -1) : input);
  };
  process.stdin.on('data', onInput);

  console.log('');
  console.log(sentence.replace(/\.+/g, '') + '.');
  collector.reset();
}

process.stdin.setRawMode(true);
display();
