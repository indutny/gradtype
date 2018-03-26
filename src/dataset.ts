import * as assert from 'assert';

export const MAX_CHAR = 27;
export const SHAPE = [ MAX_CHAR + 1, MAX_CHAR + 1, 2 ];

const MIN_STRIDE = 15;
const MAX_STRIDE = 30;
const STRIDE_STEP = 5;

const VALIDATE_PERCENT = 0.1;

const VALIDATE_MIN_STRIDE = 30;
const VALIDATE_MAX_STRIDE = 30;
const VALIDATE_STRIDE_STEP = 1;

export type DatasetEntry = ReadonlyArray<number>;

export interface IInputEntry {
  c: string;
  ts: number;
}

export interface IDatasetMeanVar {
  mean: number;
  variance: number;
}

export interface IDatasetOutput {
  train: ReadonlyArray<DatasetEntry>;
  validate: ReadonlyArray<DatasetEntry>;
}

export interface IIntermediateEntry {
  fromCode: number;
  toCode: number;
  delta: number;
}

export type Input = ReadonlyArray<IInputEntry>;
export type Intermediate = ReadonlyArray<IIntermediateEntry>;

export class Dataset {
  public generate(events: Input): Output {
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
    const out: Intermediate = [];

    let lastTS: number | undefined;
    let lastCode: number | undefined;
    for (const event of events) {
      // Skip `Enter` and things like that
      if (event.k.length !== 1) {
        lastTS = undefined;
        lastCode = undefined;
        continue;
      }

      // TODO(indutny): backspace?
      const code = this.compress(event.k.charCodeAt(0));
      if (code === undefined) {
        lastTS = undefined;
        lastCode = undefined;
        continue;
      }

      if (lastCode !== undefined) {
        out.push({
          delta: event.ts - lastTS,
          fromCode: lastCode,
          toCode: code,
        });
      }
      lastTS = event.ts;
      lastCode = code;
    }
    return out;
  }

  private stride(input: Intermediate, min: number, max: number, step: number)
    : ReadonlyArray<DatasetEntry> {
    if (input.length < 2 * max) {
      throw new Error('Not enough events to generate a stride');
    }

    const meanVar = this.computeMeanVar(input);

    const res: DatasetEntry[] = [];
    for (let stride = min; stride <= max; stride += step) {
      for (let i = 0; i < stride; i++) {
        for (let j = i; j < input.length; j += stride) {
          const slice = input.slice(j, j + stride);
          if (slice.length !== stride) {
            break;
          }

          res.push(this.single(slice, meanVar));
        }
      }
    }
    return res;
  }

  public single(input: Intermediate, meanVar?: IDatasetMeanVar): DatasetEntry {
    if (meanVar === undefined) {
      meanVar = this.computeMeanVar(input);
    }
    const mean = meanVar!.mean;
    const variance = meanVar!.variance;
    const size = (MAX_CHAR + 1) * (MAX_CHAR + 1);
    const result: number[] = new Array(2 * size).fill(0);
    const count: number[] = new Array(size).fill(0);
    for (const event of input) {
      const fromCode = event.fromCode;
      const toCode = event.toCode;
      if (fromCode === undefined || toCode === undefined) {
        continue;
      }

      assert(0 <= fromCode && fromCode <= MAX_CHAR);
      assert(0 <= toCode && toCode <= MAX_CHAR);
      const index = fromCode + toCode * (MAX_CHAR + 1);

      result[2 * index] += (event.delta - mean) / variance;
      result[2 * index + 1] = 1;
      count[index]++;
    }

    for (let i = 0; i < count.length; i++) {
      if (count[i] === 0) {
        continue;
      }

      result[2 * i] /= count[i];
    }

    return result;
  }

  private computeMeanVar(events: Intermediate): IDatasetMeanVar {
    let mean = 0;
    let variance = 0;
    for (const event of events) {
      mean += event.delta;
      variance += Math.pow(event.delta, 2);
    }
    mean /= events.length;
    variance /= events.length;
    variance = Math.sqrt(variance - Math.pow(mean, 2));

    return { mean, variance };
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
