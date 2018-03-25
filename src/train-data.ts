import * as debugAPI from 'debug';
import * as propel from 'propel';

const debug = debugAPI('gradtype:train-data');

const BULK = 128;

export interface ITrainDataBulk {
  readonly input: propel.Tensor;
  readonly labels: propel.Tensor;
}

export interface ITrainDataResult {
  readonly maxIndex: number;
  readonly bulks: ReadonlyArray<ITrainDataBulk>;
}

export class TrainData {
  public parse(csv: string): ITrainDataResult {
    const lines = csv
      .split(/\r\n|\r|\n/g).map((line) => line.trim()).filter((line) => line);

    const recordings: Tensor[] = [];
    const indices: number[] = [];
    let maxIndex: number = 0;
    while (lines.length !== 0) {
      // Consume less memory
      const line = lines.pop();
      const parts = line.split(/\s*,\s*/g);

      if (lines.length % 500 === 0) {
        debug('left to load %d', lines.length);
      }

      const single = new Float32Array(parts.length - 1);
      for (let i = 0; i < single.length; i++) {
        single[i] = parseFloat(parts[i]);
      }
      const index = parts[parts.length - 1];
      maxIndex = Math.max(maxIndex, index);

      indices.push(index);
      recordings.push(propel.tensor(single));
    }

    const result: ITrainDataBulk[] = [];
    for (let i = 0; i < indices.length; i += BULK) {
      const labels = propel.tensor(indices.slice(i, i + BULK), {
        dtype: 'int32'
      }).oneHot(maxIndex + 1);
      const input = propel.stack(recordings.slice(i, i + BULK), 0);

      result.push({ labels, input });
    }
    debug('dataset count %d', result.length);

    return { maxIndex, bulks: result };
  }
}
