import { Collector } from '../src/gradtype';

describe('gradtype/collector', () => {
  it('should register event and give result', () => {
    const c = new Collector();

    c.register(3);
    c.register(4);
    c.reset();
    c.register(3);
    c.register(6);

    c.getResult();
  });
});
