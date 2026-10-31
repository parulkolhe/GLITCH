import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PHASES, next, reduce, initialMachine, TRANSITIONS, type PhaseEvent } from '../src/game/stateMachine';
import { buildCampaign, zoneQueue, finalSequence, AUTHORED_ZONES } from '../src/game/campaign';
import { scoreRecovery, streakMultiplier, rankFor } from '../src/game/scoring';
import { newProgress, recordMiss, recordRecovery, recordSkip, completeZone, integrity } from '../src/game/progress';
import { validateProgress, loadProgress, saveProgress, loadSettings, DEFAULT_SETTINGS, type StorageLike } from '../src/game/save';
import { evaluateAchievements } from '../src/game/achievements';
import { simplify, compactStrokes, expandStrokes, inkAmount } from '../src/canvas/strokes';
import { recognizedLine, missedLine, finaleBeats, type ReactionContext } from '../src/story/dialogue';
import { CONTENT, contentFor } from '../src/story/memories';

const MODEL_CLASSES = ['cat', 'dog', 'fish', 'bird', 'house', 'tree', 'flower', 'sun', 'moon', 'star', 'car', 'bicycle', 'airplane', 'sailboat',
  'apple', 'banana', 'pizza', 'cake', 'umbrella', 'clock', 'guitar', 'chair', 'key', 'lightning', 'cloud', 'headphones', 'frog', 'duck',
  'rabbit', 'butterfly', 'mountain', 'book'];
const tiers = Object.fromEntries(MODEL_CLASSES.map((c, i) => [c, ['easy', 'medium', 'hard'][i % 3]]));

// ------------------------------------------------------------- state machine
test('happy path through one recovery follows legal transitions', () => {
  const path: PhaseEvent[] = ['BOOT_DONE', 'NEW_GAME', 'INTRO_DONE', 'CORE_ONLINE', 'ZONE_READY', 'STROKE_START', 'SUBMIT', 'RESULT_RECOGNIZED', 'RESTORE', 'NEXT_TARGET'];
  let m = initialMachine;
  for (const e of path) {
    const before = m.phase;
    m = reduce(m, e);
    assert.notEqual(m.phase, undefined);
    assert.equal(m.lastEvent, e, `${e} should be accepted in ${before}`);
  }
  assert.equal(m.phase, 'DRAWING_READY');
  assert.equal(m.rejected, 0);
});

test('impossible transitions are rejected and leave the phase unchanged', () => {
  assert.equal(next('DRAWING_READY', 'SUBMIT'), null, 'cannot submit an empty canvas');
  assert.equal(next('PREDICTION_PENDING', 'SUBMIT'), null, 'no double submit');
  assert.equal(next('PREDICTION_MISSED', 'RESTORE'), null, 'a miss can never restore a memory');
  assert.equal(next('PREDICTION_UNCERTAIN', 'RESTORE'), null, 'uncertainty never counts as success');
  assert.equal(next('MEMORY_RESTORATION', 'RESTORE'), null, 'no double restoration');
  const m = reduce({ ...initialMachine, phase: 'PREDICTION_MISSED' }, 'RESULT_RECOGNIZED');
  assert.equal(m.phase, 'PREDICTION_MISSED');
  assert.equal(m.rejected, 1);
});

test('clearing during a pending prediction returns to a blank canvas', () => {
  assert.equal(next('PREDICTION_PENDING', 'CLEAR'), 'DRAWING_READY');
});

test('every phase except BOOT can reach the title; every target phase exists', () => {
  for (const phase of PHASES) {
    if (phase !== 'BOOT' && phase !== 'TITLE') assert.equal(next(phase, 'TO_TITLE'), 'TITLE');
    for (const target of Object.values(TRANSITIONS[phase] ?? {})) assert.ok(PHASES.includes(target!));
  }
});

// ------------------------------------------------------------------ campaign
test('campaign covers every model class exactly once and drops unknown ones', () => {
  const c = buildCampaign(MODEL_CLASSES, tiers);
  const all = c.zones.flatMap(z => z.classes);
  assert.equal(all.length, MODEL_CLASSES.length);
  assert.deepEqual(new Set(all), new Set(MODEL_CLASSES));
  const small = buildCampaign(['house', 'cat', 'zebra'], { zebra: 'hard' });
  assert.ok(small.zones.flatMap(z => z.classes).includes('zebra'), 'unauthored model classes are distributed into zones');
  assert.ok(!small.zones.flatMap(z => z.classes).includes('sun'), 'classes the model lacks are removed');
  for (const z of small.zones) if (!z.final) assert.ok(z.required <= z.classes.length);
});

test('zone queue orders easy first, excludes recovered and puts skipped last', () => {
  const c = buildCampaign(MODEL_CLASSES, { ...tiers, house: 'easy', sun: 'hard', mountain: 'medium', umbrella: 'easy', star: 'medium' });
  const zone = c.zones[0];
  const q = zoneQueue(zone, c, new Set(['star']), ['umbrella'], 1);
  assert.ok(!q.includes('star'));
  assert.equal(q[q.length - 1], 'umbrella');
  assert.equal(q[0], 'house');
  assert.equal(q.indexOf('sun'), q.length - 2, 'hard tier comes after easier ones');
  assert.deepEqual(zoneQueue(zone, c, new Set(), [], 7), zoneQueue(zone, c, new Set(), [], 7), 'seeded and reproducible');
});

test('final sequence prefers skipped and struggled classes, without repeats', () => {
  const c = buildCampaign(MODEL_CLASSES, tiers);
  const seq = finalSequence(c, { recovered: { cat: { attempts: 5 }, house: { attempts: 1 }, sun: { attempts: 1 } }, skipped: ['dog'], seen: ['cat', 'dog', 'house', 'sun'] }, 3, 5);
  assert.equal(new Set(seq).size, seq.length);
  assert.ok(seq.includes('dog') && seq.includes('cat'));
  assert.equal(seq.length, 5);
});

// ------------------------------------------------------------------- scoring
test('scoring rewards recovery, never punishes misses below a floor, and is not time-based', () => {
  const first = scoreRecovery({ attempts: 1, hintsUsed: 0, streakBefore: 0, tier: 'easy', targetScore: 1 });
  assert.equal(first.points, 125);
  assert.equal(first.firstTry, true);
  assert.equal(first.streakAfter, 1);
  const late = scoreRecovery({ attempts: 9, hintsUsed: 5, streakBefore: 4, tier: 'hard', targetScore: 0.5 });
  assert.equal(late.streakAfter, 0);
  assert.ok(late.points >= Math.round(150 * 0.5), 'multiplier floor of 0.5');
  assert.equal(late.points, 75 + 13 + 20);
  assert.equal(streakMultiplier(1), 1);
  assert.equal(streakMultiplier(20), 1.5);
  const streaky = scoreRecovery({ attempts: 1, hintsUsed: 0, streakBefore: 2, tier: 'medium', targetScore: 0.9 });
  assert.equal(streaky.points, Math.round(125 * 1.2) + 23);
});

test('ranks progress with XP', () => {
  assert.equal(rankFor(0).index, 0);
  assert.equal(rankFor(400).index, 1);
  assert.equal(rankFor(1e6).nextAt, null);
});

// ------------------------------------------------------------ progress/save
test('progress records misses, recoveries, skips and zone completion', () => {
  const c = buildCampaign(MODEL_CLASSES, tiers);
  let p = newProgress(1);
  p = recordMiss(p, 'bird', 'duck', false);
  p = recordMiss(p, 'bird', 'duck', false);
  p = recordMiss(p, 'bird', 'cat', true);
  assert.deepEqual(p.confusions, [{ target: 'bird', predicted: 'duck', count: 2 }], 'uncertain guesses are not counted as confusions');
  assert.equal(p.uncertainCount, 1);
  p = recordSkip(p, 'bird');
  p = recordRecovery(p, 'bird', { zoneId: 'living-archive', attempts: 4, hintsUsed: 1, clarity: 0.8, points: 100, strokes: [[1, 2, 3, 4]], at: 1, modelVersion: 'v' }, 0, 1, false);
  assert.ok(!p.skipped.includes('bird'));
  assert.equal(p.score, 100);
  assert.ok(evaluateAchievements(p, c).includes('not-a-duck'));
  p = completeZone(p, 'boot-sector', 200, c.zones.length);
  p = completeZone(p, 'boot-sector', 200, c.zones.length);
  assert.equal(p.score, 300, 'zone bonus only once');
  assert.ok(integrity(p, c) > 0 && integrity(p, c) < 1);
});

test('saves round-trip and corrupt saves are rejected safely', () => {
  const mem = new Map<string, string>();
  const store: StorageLike = { getItem: k => mem.get(k) ?? null, setItem: (k, v) => void mem.set(k, v), removeItem: k => void mem.delete(k) };
  const p = { ...newProgress(9), score: 420, recovered: { sun: { zoneId: 'boot-sector', attempts: 1, hintsUsed: 0, clarity: 0.9, points: 120, strokes: [[1, 2, 3, 4]], at: 5, modelVersion: 'x' } } };
  assert.ok(saveProgress(p, store));
  const back = loadProgress(store)!;
  assert.equal(back.score, 420);
  assert.deepEqual(back.recovered.sun.strokes, [[1, 2, 3, 4]]);
  mem.set('glitch.campaign.v2', '{not json');
  assert.equal(loadProgress(store), null);
  assert.equal(validateProgress({ version: 1 }), null);
  assert.equal(validateProgress({ ...p, score: 'lots' })!.score, 0);
  const throwing: StorageLike = { getItem: () => { throw new Error('blocked'); }, setItem: () => { throw new Error('blocked'); }, removeItem: () => {} };
  assert.equal(loadProgress(throwing), null);
  assert.equal(saveProgress(p, throwing), false);
  assert.deepEqual(loadSettings(throwing), DEFAULT_SETTINGS);
});

// ------------------------------------------------------------------- strokes
test('stroke simplification keeps endpoints and compact strokes round-trip', () => {
  const line: [number, number][] = Array.from({ length: 50 }, (_, i) => [i * 2, 100 + (i === 25 ? 40 : 0)]);
  const s = simplify(line, 1.5);
  assert.deepEqual(s[0], line[0]);
  assert.deepEqual(s[s.length - 1], line[line.length - 1]);
  assert.ok(s.length < line.length && s.some(p => p[1] === 140));
  // closed loops (a square that ends where it starts) must keep their corners
  const square: [number, number][] = [[0, 0], [50, 0], [100, 0], [100, 50], [100, 100], [50, 100], [0, 100], [0, 50], [0, 0]];
  const sq = simplify(square, 1.5);
  assert.ok(sq.length >= 5, `closed square collapsed to ${sq.length} points`);
  for (const corner of [[100, 0], [100, 100], [0, 100]]) assert.ok(sq.some(p => p[0] === corner[0] && p[1] === corner[1]), `lost corner ${corner}`);
  const strokes = [{ points: line }];
  const back = expandStrokes(compactStrokes(strokes));
  assert.equal(back.length, 1);
  assert.ok(inkAmount(back) > 90);
});

// ------------------------------------------------------------------ dialogue
const ctx = (over: Partial<ReactionContext>): ReactionContext => ({ target: 'bird', predicted: 'duck', second: 'airplane', score: 0.8, targetScore: 0.1,
  targetRank: 2, attempt: 1, uncertain: false, streak: 0, integrity: 0.5, ...over });

test('reactions use the real prediction and never claim learning', () => {
  const pair = missedLine(ctx({}), 0);
  assert.match(pair, /duck/i);
  const unsure = missedLine(ctx({ uncertain: true, score: 0.31 }), 0);
  assert.match(unsure, /31%|duck|airplane|bird/);
  const ok = recognizedLine(ctx({ predicted: 'bird', targetScore: 0.99 }), 0);
  assert.match(ok, /bird/);
  for (const line of [pair, unsure, ok]) assert.doesNotMatch(line, /\blearn|train/i);
});

test('finale responds to actual progress', () => {
  const full = finaleBeats({ recovered: 32, total: 32, skipped: [], bestStreak: 6, topConfusion: null, clearest: null, hardestWon: null, attempts: 40 });
  const partial = finaleBeats({ recovered: 20, total: 32, skipped: ['dog'], bestStreak: 2, topConfusion: { target: 'cat', predicted: 'dog', count: 3 }, clearest: null, hardestWon: { name: 'frog', attempts: 6 }, attempts: 80 });
  assert.ok(full.some(b => /Every single one/.test(b.text)));
  assert.ok(partial.some(b => /12 are still dark/.test(b.text)));
  assert.ok(partial.some(b => /frog took 6 tries/.test(b.text)));
  assert.ok(partial.some(b => /cats looked a bit like a dog/.test(b.text)));
});

test('every authored class has a memory and two echoes', () => {
  for (const zone of AUTHORED_ZONES) for (const c of zone.classes) {
    assert.ok(CONTENT[c], `missing content for ${c}`);
    assert.equal(contentFor(c).echoes.length, 2);
  }
  assert.match(contentFor('zebra').memory, /zebra/);
});
