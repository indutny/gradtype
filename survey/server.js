'use strict';

const fs = require('fs');
const crypto = require('crypto');
const path = require('path');
const util = require('util');
const { send, json } = require('micro');

const Joi = require('joi');

const MIN_SEQUENCE_LEN = 1;
const OUT_DIR = path.join(__dirname, 'datasets');

try {
  fs.mkdirSync(OUT_DIR);
} catch(e) {
}

const Dataset = Joi.array().items(
  Joi.object().keys({
    ts: Joi.number().required(),
    k: Joi.string().required()
  })
).min(MIN_SEQUENCE_LEN);

module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', 'https://indutny.github.io');
  res.setHeader('Vary', 'Origin');
  res.setHeader('Access-Control-Allow-Methods', 'GET, PUT');

  if (req.method !== 'PUT') {
    return '';
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
  const hash = crypto.createHash('sha256').update(data).digest('hex');

  const file = path.join(OUT_DIR, hash + '.json');

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

  return 'ok';
};
