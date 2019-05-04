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
    this.index = {
      data: 0,
      sentence: 0,
      letter: 0,
    };
    this.timer = null;
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

    this.data = await this.load(uri);
    this.index = { data: 0, sentence: 0, letter: 0 };
    this.text.textContent = '';
    clearTimeout(this.timer);
    this.animate(0);
  }

  animate(delta = 0) {
    if (this.index.data === this.data.length) {
      return;
    }

    const curr = this.data[this.index.data];
    const next = this.index.data + 1 < this.data.length ?
      this.data[this.index.data + 1] :
      null;

    if (next) {
      const nextDelta = next.ts - curr.ts;
      this.timer = setTimeout(() => this.animate(nextDelta), nextDelta * 100);
    }

    const sentence = this.sentences[this.index.sentence];
    const letter = sentence && sentence[this.index.letter].toLowerCase();

    if (curr.e === 'd' && letter === curr.k.toLowerCase()) {
      const elem = document.createElement('span');
      elem.style['font-size'] = `${Math.log(delta * 1000 + Math.E) * 10}px`;
      elem.textContent = curr.k;
      this.text.appendChild(elem);

      this.index.letter++;
      if (this.index.letter === sentence.length) {
        this.index.letter = 0;
        this.index.sentence++;
        this.text.appendChild(document.createElement('br'));
      }
    }

    this.index.data++;
  }
}

const app = new App();
document.body.appendChild(app.elem);
app.start().catch((e) => {
  throw e;
});
