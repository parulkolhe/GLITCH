/**
 * Campaign progress: everything that persists between sessions. Pure functions
 * only; the React layer calls these and hands the result to save.ts.
 */
import type { Campaign } from './campaign';

export const SAVE_VERSION = 2;

export interface RecoveredMemory {
  zoneId: string;
  attempts: number;
  hintsUsed: number;
  clarity: number;          // calibrated target score at recognition
  points: number;
  strokes: number[][];      // compact strokes of the drawing that was recognised
  at: number;
  modelVersion: string;
}

export interface Confusion { target: string; predicted: string; count: number }

export interface Progress {
  version: number;
  startedAt: number;
  zoneIndex: number;                    // index into campaign.zones of the current zone
  zonesCompleted: string[];
  recovered: Record<string, RecoveredMemory>;
  skipped: string[];
  seen: string[];                       // classes ever prompted
  finalQueue: string[] | null;          // Act III sequence, fixed once generated
  finalRecovered: string[];             // recovered during Act III
  score: number;
  xp: number;
  fragments: number;
  streak: number;
  bestStreak: number;
  attempts: number;                     // drawings submitted
  hintsUsed: number;
  hintsByZone: Record<string, number>;
  confusions: Confusion[];              // the model's real mistakes this campaign
  uncertainCount: number;
  achievements: string[];
  campaignComplete: boolean;
  introSeen: boolean;
  seed: number;
  rushBest: number;
}

export function newProgress(seed = Math.floor(Math.random() * 2 ** 31)): Progress {
  return {
    version: SAVE_VERSION, startedAt: Date.now(), zoneIndex: 0, zonesCompleted: [], recovered: {}, skipped: [],
    seen: [], finalQueue: null, finalRecovered: [], score: 0, xp: 0, fragments: 0, streak: 0, bestStreak: 0,
    attempts: 0, hintsUsed: 0, hintsByZone: {}, confusions: [], uncertainCount: 0, achievements: [],
    campaignComplete: false, introSeen: false, seed, rushBest: 0,
  };
}

export function markSeen(p: Progress, target: string): Progress {
  return p.seen.includes(target) ? p : { ...p, seen: [...p.seen, target] };
}

export function recordMiss(p: Progress, target: string, predicted: string, uncertain: boolean): Progress {
  const confusions = p.confusions.map(c => ({ ...c }));
  if (!uncertain && predicted !== target) {
    const existing = confusions.find(c => c.target === target && c.predicted === predicted);
    if (existing) existing.count += 1; else confusions.push({ target, predicted, count: 1 });
  }
  return { ...p, attempts: p.attempts + 1, streak: 0, confusions, uncertainCount: p.uncertainCount + (uncertain ? 1 : 0) };
}

export function recordHint(p: Progress, zoneId: string): Progress {
  return { ...p, hintsUsed: p.hintsUsed + 1, hintsByZone: { ...p.hintsByZone, [zoneId]: (p.hintsByZone[zoneId] ?? 0) + 1 } };
}

export function recordRecovery(p: Progress, target: string, memory: RecoveredMemory, streakAfter: number, fragments: number, isFinal: boolean): Progress {
  return {
    ...p,
    attempts: p.attempts + 1,
    recovered: { ...p.recovered, [target]: memory },
    skipped: p.skipped.filter(c => c !== target),
    finalRecovered: isFinal && !p.finalRecovered.includes(target) ? [...p.finalRecovered, target] : p.finalRecovered,
    score: p.score + memory.points,
    xp: p.xp + memory.points,
    fragments: p.fragments + fragments,
    streak: streakAfter,
    bestStreak: Math.max(p.bestStreak, streakAfter),
  };
}

export function recordSkip(p: Progress, target: string): Progress {
  return { ...p, streak: 0, skipped: p.skipped.includes(target) ? p.skipped : [...p.skipped, target] };
}

export function completeZone(p: Progress, zoneId: string, bonus: number, zoneCount: number): Progress {
  if (p.zonesCompleted.includes(zoneId)) return p;
  return {
    ...p,
    zonesCompleted: [...p.zonesCompleted, zoneId],
    zoneIndex: Math.min(p.zoneIndex + 1, zoneCount - 1),
    score: p.score + bonus,
    xp: p.xp + bonus,
  };
}

/** 0 = fully corrupted, 1 = every memory restored. Drives the robot and the world. */
export function integrity(p: Progress, campaign: Campaign): number {
  const total = campaign.classes.length || 1;
  const recovered = Object.keys(p.recovered).filter(c => campaign.classes.includes(c)).length;
  const zones = campaign.zones.length || 1;
  return Math.min(1, 0.7 * (recovered / total) + 0.3 * (p.zonesCompleted.length / zones));
}

export function topConfusions(p: Progress, limit = 3): Confusion[] {
  return [...p.confusions].sort((a, b) => b.count - a.count).slice(0, limit);
}

export function zoneRecoveredCount(p: Progress, classes: string[]): number {
  return classes.filter(c => p.recovered[c]).length;
}
