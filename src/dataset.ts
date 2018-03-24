import { ICollectorResult } from './collector';
import { MAX_CHAR } from './gradtype';

const STRIDE = 64;

export type DatasetEntry = ReadonlyArray<number>;

export class Dataset {
  public generate(events: ReadonlyArray<ICollectorResult>)
    : ReadonlyArray<DatasetEntry> {
    if (events.length < 2 * STRIDE) {
      throw new Error('Not enough events to generate dataset');
    }

    const res: SingleDataset = [];
    for (let i = 0; i < STRIDE; i++) {
      for (let j = i; j < events.length; j += STRIDE) {
        const slice = events.slice(j, j + STRIDE);
        if (slice.length !== STRIDE) {
          break;
        }

        res.push(this.generateSingle(slice));
      }
    }
    return res;
  }

  public generateSingle(events: ReadonlyArray<ICollectorResult>): DatasetEntry {
    let mean = 0;
    for (const event of events) {
      mean += event.delta;
    }
    mean /= events.length;

    const size = (MAX_CHAR + 1)* (MAX_CHAR + 1);
    const result: number[] = new Array(size * 2).fill(0);
    const count: number[] = new Array(size).fill(0);
    for (const event of events) {
      const index = event.fromCode + event.toCode * MAX_CHAR;

      result[index * 2] += event.delta;
      result[index * 2 + 1] += Math.pow(event.delta, 2);
      count[index]++;
    }

    for (let i = 0; i < count.length; i++) {
      if (count[i] === 0) {
        continue;
      }

      result[i * 2] /= count[i];

      // Deviation
      result[i * 2 + 1] /= count[i];
      result[i * 2 + 1] -= Math.pow(result[i * 2], 2);
      result[i * 2 + 1] = Math.sqrt(result[i * 2 + 1]);

      // Normalize
      result[i * 2] = Math.exp(-result[i * 2] / mean);
      result[i * 2 + 1] = Math.exp(-result[i * 2 + 1] / mean);
    }

    return result;
  }
}
