import * as performance from './performance';
import { default as sentences } from './sentences';

const API_ENDPOINT = 'https://gradtype-survey.darksi.de/';
const LS_KEY = 'gradtype-survey-v1';

type LogEvent = {
  readonly ts: number;
  readonly k: string;
} | 'r';

class Application {
  private readonly log: LogEvent[] = [];
  private readonly start: number = performance.now();
  private readonly elems = {
    display: document.getElementById('display')!,
    counter: document.getElementById('counter')!,
    wrap: document.getElementById('wrap')!,
  };

  private sentenceIndex: number = 0;
  private charIndex: number = 0;

  constructor() {
    if (window.localStorage && window.localStorage.getItem(LS_KEY)) {
      this.complete();
      return;
    }

    this.displaySentence();

    window.addEventListener('keydown', (e) => {
      this.onKeyDown(e.key);
    }, true);
  }

  displaySentence() {
    const sentence = sentences[this.sentenceIndex];

    this.elems.counter.textContent =
      (sentences.length - this.sentenceIndex).toString();
    this.elems.display.innerHTML =
      '<span class=sentence-completed>' +
      sentence.slice(0, this.charIndex) +
      '</span>' +
      '<span class=sentence-pending>' +
      sentence.slice(this.charIndex)
      '</span>';
  }

  onKeyDown(key: string) {
    const now = performance.now();

    if (this.sentenceIndex === sentences.length) {
      return;
    }

    const sentence = sentences[this.sentenceIndex];
    const expected = sentence[this.charIndex];
    if (key !== expected && !(key === ' ' && expected === '␣')) {
      return;
    }

    this.log.push({ ts: now - this.start, k: key });

    this.charIndex++;
    this.displaySentence();

    if (this.charIndex !== sentence.length) {
      return;
    }

    // Next sentence
    this.charIndex = 0;
    this.sentenceIndex++;
    this.log.push('r');

    if (this.sentenceIndex === sentences.length) {
      this.elems.counter.textContent = '0';
      this.save((err, code) => {
        if (err) {
          return this.error();
        }
        this.complete(code!);
      });
      return;
    }

    this.displaySentence();
  }

  save(callback: (err?: Error, code?: string) => void) {
    const json = JSON.stringify(this.log.map((event) => {
      if (event === 'r') {
        return event;
      }
      return { ts: (event.ts - this.start) / 1000, k: event.k };
    }));

    this.elems.wrap.innerHTML =
      '<h1>Uploading, please do not close this window...</h1>';

    const xhr = new XMLHttpRequest();

    xhr.onload = () => {
      let response: any;
      try {
        response = JSON.parse(xhr.responseText);
      } catch (e) {
        return callback(e);
      }

      if (!response.code) {
        return callback(new Error('No response code'));
      }

      return callback(undefined, response.code);
    };

    xhr.onerror = (err) => {
      return callback(new Error('XHR error'));
    };

    xhr.open('PUT', API_ENDPOINT, true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.send(json);
  }

  complete(code?: string) {
    if (window.localStorage) {
      window.localStorage.setItem(LS_KEY, 'submitted');
    }
    const wrap = this.elems.wrap;
    if (code === undefined) {
      wrap.innerHTML = '<h1>Thank you for submitting survey!</h1>';
    } else {
      wrap.innerHTML = '<h1>Thank you for submitting survey!</h1>' +
        `<h1 style="color:red">Your code is ${code}</h1>`;
    }
  }

  error() {
    this.elems.wrap.innerHTML = '<h1>Server error, please retry later!</h1>';
  }
}

const app = new Application();
