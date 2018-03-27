import * as assert from 'assert';

import { shuffle } from './utils';

export const MAX_CHAR = 28;
export const SHAPE = [ MAX_CHAR + 1, MAX_CHAR + 1, 2 ];

const CUTOFF_TIME = 3;
const MOVING_AVG_WINDOW = 8;

const MIN_STRIDE = 30;
const MAX_STRIDE = 30;
const STRIDE_STEP = 1;

const VALIDATE_PERCENT = 0.25;

const VALIDATE_MIN_STRIDE = 30;
const VALIDATE_MAX_STRIDE = 30;
const VALIDATE_STRIDE_STEP = 1;

export type DatasetEntry = ReadonlyArray<number>;

export interface IInputEntry {
  k: string;
  ts: number;
}

export interface IDatasetOutput {
  train: Iterator<DatasetEntry>;
  validate: Iterator<DatasetEntry>;
}

export interface IIntermediateEntry {
  fromCode: number;
  toCode: number;
  delta: number;
}

export type Input = ReadonlyArray<IInputEntry>;
export type Intermediate = ReadonlyArray<IIntermediateEntry>;

export class Dataset {
  public generate(events: Input): IDatasetOutput {
    const ir = this.preprocess(events);

    const validateCount = Math.ceil(VALIDATE_PERCENT * ir.length);
    return {
      train: this.stride(ir.slice(validateCount),
        MIN_STRIDE, MAX_STRIDE, STRIDE_STEP),
      validate: this.stride(ir.slice(0, validateCount),
        VALIDATE_MIN_STRIDE, VALIDATE_MAX_STRIDE, VALIDATE_STRIDE_STEP),
    };
  }

  private preprocess(events: Input): Intermediate {
    const out: IIntermediateEntry[] = [];

    // Moving average
    let average = 0;
    const deltaList: number[] = [];

    let lastTS: number | undefined;
    let lastCode: number | undefined;

    const reset = () => {
      lastTS = undefined;
      lastCode = undefined;
    };

    for (const event of events) {
      // Skip `Enter` and things like that
      if (event.k.length !== 1) {
        reset();
        continue;
      }

      // TODO(indutny): backspace?
      const code = this.compress(event.k.charCodeAt(0));
      if (code === undefined || (event.ts - lastTS!) >= CUTOFF_TIME) {
        reset();
        continue;
      }

      if (lastCode !== undefined) {
        const delta = event.ts - lastTS!;

        average += delta / MOVING_AVG_WINDOW;
        if (deltaList.length >= MOVING_AVG_WINDOW - 1) {
          average -= deltaList[out.length - MOVING_AVG_WINDOW + 1];
        }
        deltaList.push(delta / MOVING_AVG_WINDOW);

        out.push({
          delta: delta / average,
          fromCode: lastCode,
          toCode: code,
        });
      }

      lastTS = event.ts;
      lastCode = code;
    }
    return out.slice(MOVING_AVG_WINDOW);
  }

  private *stride(input: Intermediate, min: number, max: number, step: number)
    : Iterator<DatasetEntry> {
    if (input.length < max) {
      throw new Error('Not enough events to generate a stride');
    }

    const strideSizes: number[] = [];
    for (let stride = min; stride <= max; stride += step) {
      strideSizes.push(stride);
    }

    shuffle(strideSizes);
    for (const stride of strideSizes) {
      const limit = Math.min(input.length - stride, stride - 1);

      const offsets: number[] = [];
      for (let i = 0; i <= limit; i++) {
        offsets.push(i);
      }

      shuffle(offsets);

      for (const offset of offsets) {
        const strideOffsets: number[] = [];
        for (let i = offset; i < input.length; i += stride) {
          strideOffsets.push(i);
        }

        shuffle(strideOffsets)

        for (const strideOffset of strideOffsets) {
          const slice = input.slice(strideOffset, strideOffset + stride);
          if (slice.length !== stride) {
            break;
          }

          yield this.single(slice);
        }
      }
    }
  }

  public single(input: Intermediate): DatasetEntry {
    const size = (MAX_CHAR + 1) * (MAX_CHAR + 1);
    const result: number[] = new Array(2 * size).fill(0);
    const count: number[] = new Array(size).fill(0);
    for (const event of input) {
      const fromCode = event.fromCode;
      const toCode = event.toCode;

      assert(0 <= fromCode && fromCode <= MAX_CHAR);
      assert(0 <= toCode && toCode <= MAX_CHAR);
      const index = fromCode + toCode * (MAX_CHAR + 1);

      result[2 * index] += event.delta - 1;
      result[2 * index + 1] = 1;
      count[index]++;
    }

    for (let i = 0; i < count.length; i++) {
      if (count[i] === 0) {
        continue;
      }

      result[2 * i] /= count[i];
    }

    assert(result.some((cell) => cell !== 0));
    return result;
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
