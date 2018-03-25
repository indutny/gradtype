import * as assert from 'assert';

import { ICollectorResult } from './collector';

const MAX_CHAR = 27;

export type DatasetEntry = ReadonlyArray<number>;

export class Dataset {
  public generate(events: ReadonlyArray<ICollectorResult>)
    : ReadonlyArray<DatasetEntry> {
    if (events.length < 2 * 64) {
      throw new Error('Not enough events to generate dataset');
    }

    const res: SingleDataset = [];
    for (let stride = 32; stride < 64; stride++) {
      for (let i = 0; i < stride; i++) {
        for (let j = i; j < events.length; j += stride) {
          const slice = events.slice(j, j + stride);
          if (slice.length !== stride) {
            break;
          }

          res.push(this.generateSingle(slice));
        }
      }
    }
    return res;
  }

  public generateSingle(events: ReadonlyArray<ICollectorResult>): DatasetEntry {
    let mean = 0;
    let variance = 0;
    for (const event of events) {
      mean += event.delta;
      variance += Math.pow(event.delta, 2);
    }
    mean /= events.length;
    variance /= events.length;
    variance = Math.sqrt(variance - Math.pow(mean, 2));

    const size = (MAX_CHAR + 1) * (MAX_CHAR + 1);
    const result: number[] = new Array(size).fill(0);
    const count: number[] = new Array(size).fill(0);
    for (const event of events) {
      const fromCode = this.compress(event.fromCode);
      const toCode = this.compress(event.toCode);
      if (fromCode === undefined || toCode === undefined) {
        continue;
      }

      assert(0 <= fromCode && fromCode <= MAX_CHAR);
      assert(0 <= toCode && toCode <= MAX_CHAR);
      const index = fromCode + toCode * (MAX_CHAR + 1);

      result[index] += (event.delta - mean) / variance;
      count[index]++;
    }

    for (let i = 0; i < count.length; i++) {
      if (count[i] === 0) {
        continue;
      }

      result[i] /= count[i];
    }

    return result;
  }

  public compress(code: number): number | undefined {
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
