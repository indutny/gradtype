import * as assert from 'assert';

const MAX_CHAR = 0x7f;

type Timestamp = [ number, number ];

interface ICollectorEvent {
  readonly timestamp: Timestamp;
  readonly code: number;
}

interface ICollectorResult {
  readonly timestamp: number;
  readonly code: number;
}

export class Collector {
  private readonly start: Timestamp = process.hrtime();
  private readonly events: (ICollectorEvent | 'reset')[] = [];

  public register(code: number): void {
    this.events.push({ timestamp: process.hrtime(this.start), code });
  }

  public reset(): void {
    this.events.push('reset');
  }

  public getResult(): ReadonlyArray<ICollectorEvent | 'reset'> {
    return this.events.map((event) => {
      if (event === 'reset') {
        return 'reset';
      } else {
        return {
          timestamp: event.timestamp[0] + event.timestamp[1] * 1e-9,
          code: event.code,
        };
      }
    });
  }
}
