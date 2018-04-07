import * as assert from 'assert';

import { shuffle } from './utils';

export const MAX_CHAR = 27;

const CUTOFF_TIME = Infinity;
const MIN_SEQUENCE = 8;

// Moving average window
const WINDOW = 7;

export interface IInputEntry {
  readonly k: string;
  readonly ts: number;
}

export interface ISequenceElem {
  readonly code: number;
  readonly delta: number;
}

export type Sequence = ReadonlyArray<ISequenceElem>;

export type Input = ReadonlyArray<IInputEntry>;
export type Output = ReadonlyArray<Sequence>;

type IntermediateEntry = 'reset' | ISequenceElem;

export class Dataset {
  public generate(events: Input): Output {
    const out: ISequenceElem[][] = [];

    let sequence: ISequenceElem[] = [];
    for (const event of this.preprocess(events)) {
      if (event === 'reset') {
        if (sequence.length > MIN_SEQUENCE) {
          out.push(sequence);
        }
        sequence = [];
        continue;
      }

      sequence.push(event);
    }
    if (sequence.length > MIN_SEQUENCE) {
      out.push(sequence);
    }

    return out;
  }

  private *preprocess(events: Input): Iterator<IntermediateEntry> {
    let lastTS: number | undefined;
    let average = 0;
    let deltaHistory: number[] = [];

    const reset = (): IntermediateEntry => {
      lastTS = undefined;
      deltaHistory = [];
      average = 0;
      return 'reset';
    };

    for (const event of events) {
      let k: string = event.k;
      if (k === 'Spacebar') {
        k = ' ';
      } else if (k === '.') {
        yield reset();
        continue;
      }

      // XXX(indutny): skip everything that we don't understand
      const code = this.compress(event.k.charCodeAt(0));
      if (code === undefined) {
        continue;
      }
      assert(0 <= code && code <= MAX_CHAR);

      let delta = event.ts - (lastTS === undefined ? event.ts : lastTS);
      if (delta > CUTOFF_TIME) {
        yield reset();
        continue;
      }

      lastTS = event.ts;

      // Skip first keystroke
      if (delta === 0) {
        continue;
      }

      // Anscombe transformation
      delta = 2 * Math.sqrt(delta + 3 / 8)

      deltaHistory.push(delta);
      average += delta;

      if (deltaHistory.length < WINDOW) {
        continue;
      }

      const first = deltaHistory.shift();
      average -= first;

      // Normalize
      const currentAvg = average / deltaHistory.length;
      delta -= currentAvg;

      yield {
        delta,
        code,
      };
    }
  }

  private compress(code: number): number | undefined {
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
    } else {
      return undefined;
    }
  }
}
