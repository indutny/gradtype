import * as wilde from '../data/wilde.txt';

interface ILogEvent {
  readonly ts: number;
  readonly c: number;
}

const text: string = wilde.toString().replace(/\s+/g, ' ');
const sentences = text.split(/\.+(?!")/g).map((line) => line.trim());

const elems = {
  display: document.getElementById('display')!,
  input: document.getElementById('input')! as HTMLInputElement,
  save: document.getElementById('save')! as HTMLButtonElement,
  download: document.getElementById('download')!,
  counter: document.getElementById('counter')!,
};

let index = Math.floor(Math.random() * (sentences.length - 1));
let counter = 0;

function next() {
  elems.display.textContent = sentences[index++] + '.';
  if (index === sentences.length) {
    index = 0;
  }

  elems.input.focus();
  elems.input.value = '';
  elems.counter.textContent = (counter++).toString();
}

const log: ILogEvent[] = [];

const ts = window.performance === undefined ? () => Date.now() :
  () => window.performance.now();

const start = ts();
elems.input.onkeypress = (e: KeyboardEvent) => {
  log.push({ ts: ts(), c: e.keyCode });

  if (e.keyCode === 46) {
    next();
    e.preventDefault();
    return false;
  }
};

elems.save.onclick = (e: Event) => {
  e.preventDefault();

  const json = JSON.stringify(log.map((event) => {
    return { ts: (event.ts - start) / 1000, c: event.c };
  }));

  const blob = new File([ json ], 'gradtype.json', {
    type: 'application/json',
  });

  const url = URL.createObjectURL(blob);

  const a = document.createElement('a') as HTMLAnchorElement;
  a.href = url;
  a.download = 'gradtype.json';
  a.style.display = 'none';

  elems.download.textContent = '';
  elems.download.appendChild(a);

  a.click();

  URL.revokeObjectURL(url);
};

next();
