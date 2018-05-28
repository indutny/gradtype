import * as performance from './performance';
import { default as sentences } from './sentences';

const API_ENDPOINT = 'https://gradtype-survey.darksi.de/';
const LS_KEY = 'gradtype-survey-v1';

const REASSURE: string[] = [
  'You\'re doing great!',
  'Just few more!',
  'Almost there!'
];

const REASSURE_EVERY = Math.floor(sentences.length) / (REASSURE.length + 1);

type LogKind = 'd' | 'u';

type LogEvent = {
  readonly e: LogKind;
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
    reassure: document.getElementById('reassure')!,
  };

  private sentenceIndex: number = 0;
  private charIndex: number = 0;
  private lastReassure: number = 0;

  constructor() {
    if (window.localStorage && window.localStorage.getItem(LS_KEY)) {
      this.complete();
      return;
    }

    this.displaySentence();

    window.addEventListener('keydown', (e) => {
      e.preventDefault();
      this.onKeyDown(e.key);
      return false;
    }, true);

    window.addEventListener('keyup', (e) => {
      e.preventDefault();
      this.onKeyUp(e.key);
      return false;
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

  nextSentence() {
    this.charIndex = 0;
    this.sentenceIndex++;
    this.log.push('r');

    if (this.sentenceIndex - this.lastReassure >= REASSURE_EVERY) {
      this.lastReassure = this.sentenceIndex;
      this.elems.reassure.textContent = REASSURE.shift() || '';
    }

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

  onKeyDown(key: string) {
    const now = performance.now();
    this.log.push({ e: 'd', ts: (now - this.start) / 1000, k: key });

    if (this.sentenceIndex === sentences.length) {
      return;
    }

    const sentence = sentences[this.sentenceIndex];
    const expected = sentence[this.charIndex];
    if (key !== expected && !(key === ' ' && expected === '␣')) {
      return;
    }

    this.charIndex++;
    this.displaySentence();

    if (this.charIndex !== sentence.length) {
      return;
    }

    // Give enough time to record the last keystroke
    setTimeout(() => {
      this.nextSentence();
    }, 50);
  }

  onKeyUp(key: string) {
    const now = performance.now();
    this.log.push({ e: 'u', ts: (now - this.start) / 1000, k: key });
  }

  save(callback: (err?: Error, code?: string) => void) {
    const json = JSON.stringify(this.log);

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
