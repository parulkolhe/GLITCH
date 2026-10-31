import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PredictionClient, PredictionError, parsePrediction } from '../src/services/predictionClient';

const okBody = (requestId: string, top = 'star') => ({
  predictions: [{ class_name: top, score: 0.9 }, { class_name: 'sun', score: 0.05 }],
  uncertain: false, uncertain_below: 0.69, entropy: 0.1, margin: 0.85, target: 'star', target_score: 0.9, target_rank: 1,
  attention: [[0, 1], [1, 0]], attention_focus: 'the centre of the drawing', model_input: 'data:image/png;base64,AAAA',
  model_version: 'baseline-v1', temperature: 0.9, confidence_type: 'temperature_scaled_softmax', inference_ms: 2.5, request_id: requestId,
});

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

test('a valid response is parsed into typed fields', async () => {
  const client = new PredictionClient('', async (_u, init) => json(okBody(JSON.parse(String(init!.body)).request_id)));
  const r = await client.predict('data:image/png;base64,xx', 'star');
  assert.equal(r.predictions[0].className, 'star');
  assert.equal(r.targetRank, 1);
  assert.equal(r.modelVersion, 'baseline-v1');
  assert.equal(client.pendingId, null);
});

test('a newer request supersedes an older one; the stale answer is discarded', async () => {
  const resolvers: Array<(r: Response) => void> = [];
  const client = new PredictionClient('', (_u, init) => new Promise(resolve => {
    const id = JSON.parse(String(init!.body)).request_id;
    resolvers.push(() => resolve(json(okBody(id, resolvers.length === 1 ? 'old' : 'new'))));
    init!.signal?.addEventListener('abort', () => resolve(json(okBody(id, 'old'))));
  }));
  const first = client.predict('a', 'star');
  const second = client.predict('b', 'star');
  resolvers[1](new Response());
  await assert.rejects(first, (e: PredictionError) => e.kind === 'aborted');
  const r = await second;
  assert.equal(r.predictions[0].className, 'new');
});

test('cancel() aborts the in-flight request', async () => {
  const client = new PredictionClient('', (_u, init) => new Promise((_res, rej) => init!.signal!.addEventListener('abort', () => rej(new Error('aborted')))));
  const p = client.predict('a', null);
  client.cancel();
  await assert.rejects(p, (e: PredictionError) => e.kind === 'aborted');
});

test('server errors map to typed errors and never to a prediction', async () => {
  const c422 = new PredictionClient('', async () => json({ detail: 'The drawing is blank; add a mark before submitting.' }, 422));
  await assert.rejects(c422.predict('a', null), (e: PredictionError) => e.kind === 'invalid-input' && /blank/.test(e.message));
  const c503 = new PredictionClient('', async () => json({ detail: 'Model is unavailable.' }, 503));
  await assert.rejects(c503.predict('a', null), (e: PredictionError) => e.kind === 'unavailable');
  const down = new PredictionClient('', async () => { throw new TypeError('Failed to fetch'); });
  await assert.rejects(down.predict('a', null), (e: PredictionError) => e.kind === 'network');
});

test('timeouts surface as timeout errors', async () => {
  const client = new PredictionClient('', (_u, init) => new Promise((_res, rej) => init!.signal!.addEventListener('abort', () => rej(new Error('x')))), 20);
  await assert.rejects(client.predict('a', null), (e: PredictionError) => e.kind === 'timeout');
});

test('malformed responses are rejected', () => {
  assert.throws(() => parsePrediction({ predictions: [] }, 'r', 1), PredictionError);
  assert.throws(() => parsePrediction({ ...okBody('r'), predictions: [{ class_name: 'a', score: 0.2 }, { class_name: 'b', score: 0.7 }] }, 'r', 1), /ranked/);
  assert.throws(() => parsePrediction({ ...okBody('r'), predictions: [{ class_name: 'a', score: 1.5 }] }, 'r', 1), PredictionError);
  assert.throws(() => parsePrediction(okBody('other'), 'r', 1), /different request/);
  assert.throws(() => parsePrediction({ ...okBody('r'), uncertain: 'no' }, 'r', 1), /incomplete/);
});
