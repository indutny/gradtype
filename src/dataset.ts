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
  readonly delta: number;
}

export type Sequence = ReadonlyArray<ISequenceElem>;

export type Input = ReadonlyArray<InputEntry>;
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

  public *preprocess(events: Input): Iterator<IntermediateEntry> {
    let lastTS: number | undefined;
    let deltaHistory: number[] = [];

    const reset = (): IntermediateEntry => {
      lastTS = undefined;
      deltaHistory = [];
      return 'reset';
    };

    for (const event of events) {
      if (event === 'r') {
        yield reset();
        continue;
      }

      let k: string = event.k;

      let code: number = 0;
      try {
        code = this.compress(event.k.charCodeAt(0));
      } catch (e) {
        continue;
      }
      assert(0 <= code && code <= MAX_CHAR);

      let delta = event.ts - (lastTS === undefined ? event.ts : lastTS);
      lastTS = event.ts;

      // Skip first keystroke
      if (delta === 0) {
        continue;
      }

      yield {
        type: event.e === 'd' ? 1 : -1,
        delta,
        code,
      };
    }
  }

  public *port(events: Input): Iterator<Input> {
    for (const event of events) {
      if (event.k === '.') {
        yield 'r';
        continue;
      }

      let k: string = event.k;
      if (k === 'Spacebar') {
        k = ' ';
      }

      try {
        this.compress(k.charCodeAt(0));
      } catch (e) {
        continue;
      }

      yield { k, ts: event.ts };
    }
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
      throw new Error('Unexpected code: ' + code);
    }
  }
}
