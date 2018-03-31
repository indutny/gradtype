import * as wilde from '../data/wilde.txt';
import leven = require('leven');

const API_ENDPOINT = 'https://gradtype-survey.darksi.de/';
const LS_KEY = 'gradtype-survey-v1';
const INITIAL_COUNTER = 23;
const TOLERANCE = 0.5;

const elems = {
  display: document.getElementById('display')!,
  input: document.getElementById('input')! as HTMLInputElement,
  download: document.getElementById('download')!,
  counter: document.getElementById('counter')!,
  wrap: document.getElementById('wrap')!,
};

interface ILogEvent {
  readonly ts: number;
  readonly k: string;
}

const text: string = wilde.toString().replace(/\s+/g, ' ');
const sentences = text.split(/\.+/g)
  .filter((line) => !/["?!]/.test(line))
  .map((line) => line.trim())
  .filter((line) => line.length > 15);

let index = Math.floor(Math.random() * (sentences.length - 1));

let counter = INITIAL_COUNTER;
elems.counter.textContent = counter.toString();

const log: ILogEvent[] = [];

function next() {
  const prior = elems.display.textContent || '';
  const sentence = sentences[index++];
  elems.display.textContent = sentence + '.';
  if (index === sentences.length) {
    index = 0;
  }

  const entered = elems.input.value;
  if (prior !== '') {
    const distance = leven(entered, prior);

    // Remove last sentence
    if (distance > TOLERANCE * prior.length) {
      let i: number = 0;
      for (i = log.length - 1; i >= 0; i--) {
        if (log[i].k === '.') {
          break;
        }
      }

      log.splice(i, log.length - i);
      counter++;
    }
  }

  elems.input.focus();
  elems.input.value = '';
  elems.counter.textContent = (--counter).toString();

  if (counter === 0) {
    save();
  }
}

const ts = window.performance === undefined ? () => Date.now() :
  () => window.performance.now();

const start = ts();
elems.input.onkeypress = (e: KeyboardEvent) => {
  log.push({ ts: ts(), k: e.key });

  if (e.key === '.') {
    next();
    e.preventDefault();
    return;
  }
};

function save() {
  const json = JSON.stringify(log.map((event) => {
    return { ts: (event.ts - start) / 1000, k: event.k };
  }));

  elems.wrap.innerHTML =
    '<h1>Uploading, please do not close this window...</h1>';

  const xhr = new XMLHttpRequest();

  xhr.onload = () => {
    let response: any;
    try {
      response = JSON.parse(xhr.responseText);
    } catch (e) {
      error();
      return;
    }

    if (!response.code) {
      error();
      return;
    }

    complete(response.code);
  };

  xhr.onerror = () => {
    error();
  };

  xhr.open('PUT', API_ENDPOINT, true);
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.send(json);
};

function complete(code?: string) {
  if (window.localStorage) {
    window.localStorage.setItem(LS_KEY, 'submitted');
  }
  if (code === undefined) {
    elems.wrap.innerHTML = '<h1>Thank you for submitting survey!</h1>';
  } else {
    elems.wrap.innerHTML = '<h1>Thank you for submitting survey!</h1>' +
      `<h1 style="color:red">Your code is ${code}</h1>`;
  }
}

function error() {
  elems.wrap.innerHTML = '<h1>Server error, please retry later!</h1>';
}

if (window.localStorage && window.localStorage.getItem(LS_KEY)) {
  complete();
} else {
  next();
}
