'use strict';

const fs = require('fs');
const https = require('https');
const Buffer = require('buffer').Buffer;
const crypto = require('crypto');
const path = require('path');
const util = require('util');
const { run, send, json } = require('micro');

const Joi = require('joi');

const MIN_SEQUENCE_LEN = 2000;
const OUT_DIR = path.join(__dirname, 'datasets');
const KEY_FILE = process.env.KEY_FILE;
const CERT_FILE = process.env.CERT_FILE;
const HMAC_KEY = Buffer.from(process.env.HMAC_KEY, 'hex');

const options = {
  key: fs.readFileSync(KEY_FILE),
  cert: fs.readFileSync(CERT_FILE)
};

const microHttps = fn => https.createServer(options, (req, res) => {
  return run(req, res, fn);
});

try {
  fs.mkdirSync(OUT_DIR);
} catch(e) {
}

const Dataset = Joi.array().items(
  Joi.alternatives().try([
    Joi.object().keys({
      e: Joi.string().required(),
      ts: Joi.number().required(),
      k: Joi.string().required()
    }),
    Joi.string().valid('r')
  ])
).min(MIN_SEQUENCE_LEN);

const server = microHttps(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', 'https://indutny.github.io');
  res.setHeader('Vary', 'Origin');
  res.setHeader('Access-Control-Allow-Methods', 'GET, PUT');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'GET' && req.url === '') {
    return '';
  }

  if (req.method === 'GET' && /^\/check\//.test(req.url)) {
    const match = req.url.match(/^\/check\/([a-z0-9]{64,64})$/);
    if (match === null) {
      send(res, 400, { error: 'invalid check' });
      return;
    }

    const file = path.join(OUT_DIR, match[1] + '.json');
    const exists = await util.promisify(fs.exists)(file);
    if (exists) {
      return { ok: 'found' };
    }

    send(res, 404, { error: 'check not found' });
    return;
  }

  if (req.method !== 'PUT') {
    send(res, 200, { ok: true });
    return;
  }

  let data;
  try {
    data = await json(req, { limit: '1mb', encoding: 'utf8' });
  } catch (e) {
    send(res, 400, { error: 'Invalid JSON' });
    return;
  }

  const { error, value } = Joi.validate(data, Dataset);
  if (error) {
    send(res, 400, { error: error.message });
    return;
  }

  data = JSON.stringify(value);
  const hash = crypto.createHmac('sha256', HMAC_KEY).update(data).digest('hex');

  const file = path.join(OUT_DIR, hash + '.json');
  const meta = path.join(OUT_DIR, hash + '.meta.json');

  const exists = await util.promisify(fs.exists)(file);
  if (exists) {
    send(res, 400, { error: 'duplicate' });
    return;
  }

  try {
    await util.promisify(fs.writeFile)(file, data);
  } catch (e) {
    send(res, 500, { error: 'internal error' });
    return;
  }

  try {
    await util.promisify(fs.writeFile)(meta, JSON.stringify({
      headers: req.headers
    }));
  } catch (e) {
    send(res, 500, { error: 'internal error' });
    return;
  }

  send(res, 200, { ok: true, code: hash });
  return;
});

server.listen(process.env.PORT, 1443);
