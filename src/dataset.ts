import * as assert from 'assert';

import { shuffle } from './utils';

export const MAX_CHAR = 28;

const SAMPLE_INTERVAL = 1 / 128;  // 1 / 128 of second

export type InputEntry = {
  readonly e: 'u' | 'd';
  readonly k: string;
  readonly ts: number;
} | 'r';

export type SequenceRow = ReadonlyArray<number>;
export type Sequence = ReadonlyArray<SequenceRow>;

export type Input = ReadonlyArray<InputEntry>;
export type Output = ReadonlyArray<Sequence>;

type IntermediateEntry = 'reset' | SequenceRow;

export class Dataset {
  public generate(events: Input): Output {
    const out: ISequenceElem[][] = [];

    let sequence: ISequenceElem[] = [];
    for (const row of this.preprocess(events)) {
      if (event === 'reset') {
        sequence = [];
        continue;
      }

      sequence.push(row);
    }
    out.push(sequence);

    return out;
  }

  public *preprocess(events: Input): Iterator<IntermediateEntry> {
    let ts: number | undefined;
    let row: SequenceRow = new Array(MAX_CHAR + 1).fill(0);

    const reset = (): IntermediateEntry => {
      ts = undefined;
      return 'reset';
    };

    for (const event of events) {
      if (event === 'r') {
        yield reset();
        continue;
      }

      let k: string = event.k;

      let code: number;
      try {
        code = this.compress(event.k.charCodeAt(0));
      } catch (e) {
        continue;
      }
      assert(0 <= code && code <= MAX_CHAR);

      if (ts === undefined) {
        ts = event.ts;
      }

      while (event.ts - ts > SAMPLE_INTERVAL) {
        yield row.slice();
        ts += SAMPLE_INTERVAL;
      }

      if (event.e === 'u') {
        row[code] = 0;
      } else {
        row[code] = 1;
      }
    }

    // Final row
    if (row.some(e => e != 0)) {
      yield row.slice();
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
