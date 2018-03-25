import * as assert from 'assert';

const MAX_CHAR = 0x7f;

type Timestamp = [ number, number ];

interface ICollectorEvent {
  readonly timestamp: Timestamp;
  readonly code: number;
}

export interface ICollectorResult {
  readonly delta: number;
  readonly fromCode: number;
  readonly fromTS: Timestamp;
  readonly toCode: number;
  readonly toTS: Timestamp;
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

      const delta = (event.timestamp[0] - lastEvent.timestamp[0]) +
        1e-9 * (event.timestamp[1] - lastEvent.timestamp[1]);

      result.push({
        fromTS: lastEvent.timestamp,
        toTS: event.timestamp,
        delta,
        fromCode: lastEvent.code,
        toCode: event.code,
      });
      lastEvent = event;
    }
    return result;
  }
}
