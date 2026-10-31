// Run with: node --test tests/agents.test.js
const test = require('node:test');
const assert = require('node:assert/strict');
const A = require('../agents.js');

const tiers = { house: 'easy', apple: 'easy', star: 'medium', fish: 'medium', cat: 'hard', bird: 'hard' };
const preds = (...rows) => rows.map(([label, confidence]) => ({ label, confidence }));
const hit = A.outcome(preds(['cat', 0.9], ['dog', 0.05], ['rabbit', 0.02]), 'cat');
const weakHit = A.outcome(preds(['cat', 0.45], ['dog', 0.4], ['rabbit', 0.1]), 'cat');
const close = A.outcome(preds(['dog', 0.5], ['cat', 0.3], ['rabbit', 0.1]), 'cat');
const miss = A.outcome(preds(['house', 0.7], ['car', 0.2], ['book', 0.05]), 'cat');

test('outcome reads rank and confidence of the target', () => {
  assert.equal(hit.topOne, true);
  assert.equal(close.topOne, false);
  assert.equal(close.topThree, true);
  assert.equal(close.targetConfidence, 0.3);
  assert.equal(miss.topThree, false);
  assert.equal(miss.targetConfidence, 0);
});

test('difficulty starts easy, rises after a confident hit, falls after a miss', () => {
  const agent = A.createDifficultyAgent(tiers, () => 0);
  assert.equal(agent.level, 'easy');
  assert.equal(agent.nextTarget().tier, 'easy');
  assert.equal(agent.update(hit), 'medium');
  assert.equal(agent.update(weakHit), 'medium', 'a low-confidence hit does not raise difficulty');
  assert.equal(agent.update(hit), 'hard');
  assert.equal(agent.update(hit), 'hard', 'capped at hard');
  assert.equal(agent.update(close), 'hard', 'top-3 keeps the level');
  assert.equal(agent.update(miss), 'medium');
});

test('difficulty never repeats a prompt and falls back to a neighbouring tier', () => {
  const agent = A.createDifficultyAgent(tiers, () => 0);
  const seen = new Set();
  for (let i = 0; i < 6; i++) {
    const pick = agent.nextTarget();
    assert.ok(pick, 'a prompt is available');
    assert.ok(!seen.has(pick.label), 'no repeats');
    seen.add(pick.label);
  }
  assert.equal(agent.nextTarget(), null, 'returns null once every prompt is used');
});

test('timer shrinks after a fast confident win, grows after a miss, stays in 15-40 s', () => {
  const timer = A.createTimerAgent();
  assert.equal(timer.seconds, 30);
  assert.equal(timer.update(hit, 20), 25);
  assert.equal(timer.update(hit, 5), 25, 'a slow win does not shrink time');
  for (let i = 0; i < 10; i++) timer.update(hit, 40);
  assert.equal(timer.seconds, 15);
  for (let i = 0; i < 10; i++) timer.update(miss, 0);
  assert.equal(timer.seconds, 40);
});

test('score rewards confidence, difficulty and speed', () => {
  assert.equal(A.score(miss, 'hard', 30), 0);
  assert.ok(A.score(hit, 'easy', 10) > A.score(weakHit, 'easy', 10), 'confidence matters');
  assert.ok(A.score(hit, 'hard', 10) > A.score(hit, 'easy', 10), 'difficulty matters');
  assert.ok(A.score(hit, 'easy', 20) > A.score(hit, 'easy', 5), 'time left matters');
  assert.ok(A.score(close, 'easy', 10) < A.score(weakHit, 'easy', 10), 'top-3 earns less than top-1');
  assert.equal(A.score(hit, 'easy', 0), Math.round(100 * (0.5 + 0.45)));
});

test('feedback names the confusion and gives a class-specific hint', () => {
  assert.equal(A.feedback(hit, 'cat').tone, 'great');
  assert.equal(A.feedback(weakHit, 'cat').tone, 'ok');
  const closeFeedback = A.feedback(close, 'cat');
  assert.equal(closeFeedback.tone, 'close');
  assert.match(closeFeedback.message, /whiskers/);
  const unsure = A.feedback(A.outcome(preds(['moon', 0.2], ['banana', 0.18], ['key', 0.1]), 'cat'), 'cat');
  assert.equal(unsure.tone, 'unsure');
  assert.equal(A.feedback(miss, 'cat').tone, 'miss');
  assert.equal(A.hintFor('duck', 'bird'), A.hintFor('bird', 'duck'), 'hints are order-independent');
});
