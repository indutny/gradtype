const ts = window.performance === undefined ? () => Date.now() :
  () => window.performance.now();

export function now(): number {
  return ts();
}

export function detect(): boolean {
  const scope: any = {};

  function busy(times: number): void {
    for (let i = 0; i < times; i++) {
      if (scope.x === undefined) {
        scope.x = 0;
      } else {
        scope.x++;
      }
    }
  }

  function measure(times: number): number {
    const start = now();
    busy(times);
    return now() - start;
  }

  function mean(times: number): number {
    let result: number = 0;
    const count = 100;
    for (let i = 0; i < count; i++) {
      result += measure(times);
    }
    return result /= count;
  }

  const mul = 1.1;

  const values = [];
  for (let i = 1; i < 33554432; i *= mul) {
    const m = mean(i);
    values.push(m);
    if (m > 1) {
      break;
    }
  }

  const deltas: number[] = [];
  for (let i = values.length - 1; i >= 1; i--) {
    const cur = values[i];
    const prev = values[i - 1];

    const expected = cur / mul;
    const delta = Math.abs(expected - prev) / (expected + 1e-24);

    if (delta > 1) {
      break;
    }

    deltas.push(delta);
  }

  const percent = deltas.length / (values.length - 1);

  // Completely arbitrary
  return percent > 0.6;
}
