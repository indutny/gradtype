import * as assert from 'assert';

import { shuffle } from './utils';

export const MAX_CHAR = 28;

const MIN_SEQUENCE = 8;

// Moving average window
const WINDOW = 7;

export type InputEntry = {
  readonly e: 'u' | 'd';
  readonly k: string;
  readonly ts: number;
} | 'r';

export interface ISequenceElem {
  readonly type: number;
  readonly code: number;
  readonly duration: number;
}

export type Sequence = ReadonlyArray<ISequenceElem>;

export type Input = ReadonlyArray<InputEntry>;
export type Output = ReadonlyArray<Sequence>;

type IntermediateEntry = 'reset' | 'invalid' | ISequenceElem;

export class Dataset {
  constructor(private readonly sentences: string[]) {
  }

  public generate(events: Input): Output {
    const filtered: InputEntry[] = [];

    // Filter events
    const index = { sentence: 0, letter: 0 };
    const pressed = new Set();
    for (const event of events) {
      // TODO(indutny): do we care?
      if (event === 'r') {
        index.letter = 0;
        index.sentence++;
        pressed.clear();
        filtered.push('r');
        continue;
      }

      const key = event.k;
      const kind = event.e;

      const sentence = this.sentences[index.sentence] || '';
      const letter = sentence[index.letter] || null;

      if (kind === 'd' && key === letter) {
        if (/^[a-zA-Z ,\.]$/.test(key)) {
          pressed.add(key);
          filtered.push(event);
        }

        index.letter++;
      } else if (kind === 'u' && pressed.has(key)) {
        pressed.delete(key);
        filtered.push(event);
      }
    }

    // Just a safe-guard
    if (filtered[filtered.length - 1] !== 'r') {
      filtered.push('r');
    }

    const out: ISequenceElem[][] = [];
    let sequence: ISequenceElem[] = [];
    for (const [ i, event ] of filtered.entries()) {
      if (event === 'r') {
        if (sequence.length !== 0) {
          out.push(sequence);
        }
        sequence = [];
        continue;
      }

      let next = filtered[i + 1];
      if (next === 'r') {
        next = filtered[i + 2] || event;
      }

      if (next === 'r') {
        throw new Error('Unexpected double `r`');
      }

      const duration = next.ts - event.ts;

      sequence.push({
        type: event.e === 'd' ? 1 : -1,
        code: this.compress(event.k.charCodeAt(0)),
        duration,
      });
    }
    if (sequence.length !== 0) {
      out.push(sequence);
      sequence = [];
    }

    return out;
  }

  private compress(code: number): number {
    // 'abcdefghijklmnopqrstuvwxyz ,.'
    // a - z
    if (0x61 <= code && code <= 0x7a) {
      return code - 0x61;

    // A - Z
    } else if (0x41 <= code && code <= 0x5a) {
      return code - 0x41;

    // ' '
    } else if (code === 0x20) {
      return 26;

    // ','
    } else if (code === 0x2c) {
      return 27;

    // '.'
    } else if (code === 0x2e) {
      return 28;
    } else {
      throw new Error('Unexpected code: ' + code.toString(16));
    }
  }
}
