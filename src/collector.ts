import * as assert from 'assert';

import { MAX_CHAR } from './gradtype';

type Timestamp = [ number, number ];

interface ICollectorEvent {
  readonly timestamp: Timestamp;
  readonly code: number;
}

export interface ICollectorResult {
  readonly delta: number;
  readonly fromCode: number;
  readonly toCode: number;
}

export class Collector {
  private readonly events: (ICollectorEvent | undefined)[] = [];

  public register(code: number): void {
    this.events.push({ timestamp: process.hrtime(), code });
  }

  public reset(): void {
    this.events.push(undefined);
  }

  public getResult(): ReadonlyArray<ICollectorResult> {
    const result: ICollectorResult[] = [];

    let lastEvent: ICollectorEvent | undefined;
    for (const event of this.events) {
      // reset
      if (event === undefined) {
        lastEvent = undefined;
        continue;
      }

      if (lastEvent === undefined) {
        lastEvent = event;
        continue;
      }

      assert(event.code <= MAX_CHAR, 'Only ASCII chars are supported');
      const delta = (event.timestamp[0] - lastEvent.timestamp[0]) +
        1e-9 * (event.timestamp[1] - lastEvent.timestamp[1]);

      result.push({ delta, fromCode: lastEvent.code, toCode: event.code });
      lastEvent = event;
    }
    return result;
  }
}
