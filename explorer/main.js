class App {
  constructor() {
    this.elem = document.createElement('div');
    this.select = document.createElement('select');
    this.text = document.createElement('section');

    this.elem.appendChild(this.select);
    this.elem.appendChild(this.text);

    this.select.addEventListener('change', (e) => {
      this.onSelect(this.select.value).catch((e) => {
        throw e;
      });
    });

    this.sentences = [];
    this.data = [];
    this.lastTS = 0;
    this.index = 0;
    this.subIndex = 0;
    this.timer = null;
    this.active = new Map();
  }

  async load(uri) {
    const res = await fetch(uri);
    try {
      return await res.json();
    } catch (e) {
      console.error(`Failed to load: ${uri}`);
      throw e;
    }
  }

  async start() {
    const index = await this.load('/datasets/index.json');
    this.sentences = await this.load('/datasets/sentences.json');

    const empty = document.createElement('option');
    this.select.appendChild(empty);

    for (const file of index) {
      const option = document.createElement('option');
      option.value = '/datasets/' + file + '.json';
      option.textContent = file;

      this.select.appendChild(option);
    }
  }

  async onSelect(uri) {
    if (!uri) {
      return;
    }

    const data = await this.load(uri);
    if (data.version === 2) {
      this.index = 0;
      this.subIndex = 0;
      this.data = data.sequences;
      this.text.textContent = '';

      clearTimeout(this.timer);
      this.animateV2();
      return;
    }

    const index = { sentence: 0, letter: 0 };

    const pressed = new Set();

    this.data = [];
    data.forEach((event, i) => {
      const sentence = this.sentences[index.sentence];
      const letter = sentence && sentence[index.letter].toLowerCase();
      const key = (event.k || '').toLowerCase();
      if (event.e === 'd' && key === letter) {
        index.letter++;
        this.data.push(event);
        if (index.letter === sentence.length) {
          index.letter = 0;
          index.sentence++;
          this.data.push({ ts: event.ts, k: null, e: null });
          pressed.clear();
        }
        pressed.add(key);
      } else if (event.e === 'u' && pressed.has(key)) {
        pressed.delete(key);
        this.data.push(event);
      }
    });

    this.active = new Map();
    this.lastTS = 0;
    this.index = 0;
    this.text.textContent = '';
    clearTimeout(this.timer);
    this.animate();
  }

  animate() {
    if (this.index === this.data.length) {
      return;
    }

    const curr = this.data[this.index];
    const next = this.index + 1 < this.data.length ?
      this.data[this.index + 1] :
      null;

    if (next) {
      const nextDelta = next.ts - curr.ts;
      this.timer = setTimeout(() => this.animate(), nextDelta * 1000);
    }

    this.index++;

    if (curr.e === 'u') {
      const key = curr.k.toLowerCase();
      const active = this.active.get(key);
      if (active) {
        active.style.color = 'black';
      }
      this.active.delete(key);
      return;
    }

    const delta = curr.ts - this.lastTS;
    this.lastTS = curr.ts;

    if (curr.k === null) {
      this.text.appendChild(document.createElement('br'));
      this.active.forEach((span) => span.style.color = 'red');
      this.active = new Map();
    } else {
      const key = curr.k.toLowerCase();

      const span = document.createElement('span');
      span.style['font-size'] = `${Math.log(delta * 1000 + Math.E) * 3}px`;
      span.style.color = 'green';
      span.textContent = key;
      this.text.appendChild(span);

      this.active.set(key, span);
    }
  }

  animateV2() {
    if (this.index === this.data.length) {
      this.text.appendChild(document.createElement('br'));
      return;
    }

    const seq = this.data[this.index];
    if (this.subIndex === seq.length) {
      this.index++;
      this.subIndex = 0;
      this.text.appendChild(document.createElement('br'));

      return this.animateV2();
    }

    const event = seq[this.subIndex];
    this.subIndex++;

    const key = this.decompress(event.code);

    const span = document.createElement('span');
    span.style['font-size'] =
      `${Math.log(event.duration * 1000 + Math.E) * 3}px`;
    span.style.color = 'green';
    span.textContent = key;
    this.text.appendChild(span);
    setTimeout(() => span.style.color = 'black', event.hold * 1000);

    this.timer = setTimeout(() => this.animateV2(), event.duration * 1000);
  }

  decompress(code) {
    // 'abcdefghijklmnopqrstuvwxyz ,.'
    // a - z
    if (0 <= code && code < 26) {
      code += 0x61;

    // ' '
    } else if (code === 26) {
      code = 0x20;

    // ','
    } else if (code === 27) {
      code = 0x2c;

    // '.'
    } else if (code === 28) {
      code = 0x2e;
    } else {
      throw new Error('Unexpected code: ' + code);
    }

    return String.fromCharCode(code);
  }
}

const app = new App();
document.body.appendChild(app.elem);
app.start().catch((e) => {
  throw e;
});
