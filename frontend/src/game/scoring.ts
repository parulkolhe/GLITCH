/**
 * Scoring rewards recovery and persistence, never drawing speed.
 *
 *   base 100 + tier bonus (medium +25, hard +50)
 *   x attempt factor (1st 1.0, 2nd 0.85, 3rd 0.7, later 0.6), minus 0.15 per echo (hint), floor 0.5
 *   x streak multiplier (+10% per consecutive first-try recovery, capped at +50%)
 *   + signal clarity: up to 25 points from the CNN's calibrated score for the target
 *   + persistence bonus: 20 points for recovering after 4 or more attempts
 *
 * Misses cost nothing. Clarity uses the model's own output for the target class;
 * it can only add points to a drawing that was already recognised as the target.
 */
import type { Tier } from './campaign';

export interface RecoveryInput {
  attempts: number;     // including the successful one, >= 1
  hintsUsed: number;
  streakBefore: number; // consecutive first-try recoveries before this one
  tier: Tier;
  targetScore: number;  // calibrated score of the target class, 0..1
}

export interface ScoreLine { label: string; value: string }
export interface RecoveryScore { points: number; fragments: number; streakAfter: number; firstTry: boolean; lines: ScoreLine[] }

const TIER_BONUS: Record<Tier, number> = { easy: 0, medium: 25, hard: 50 };
const ATTEMPT_FACTOR = [1, 0.85, 0.7];

export function attemptFactor(attempts: number): number {
  return ATTEMPT_FACTOR[Math.max(0, attempts - 1)] ?? 0.6;
}

export function streakMultiplier(streak: number): number {
  return 1 + 0.1 * Math.min(Math.max(streak - 1, 0), 5);
}

export function scoreRecovery(input: RecoveryInput): RecoveryScore {
  const attempts = Math.max(1, Math.floor(input.attempts));
  const firstTry = attempts === 1 && input.hintsUsed === 0;
  const streakAfter = firstTry ? input.streakBefore + 1 : 0;
  const base = 100 + TIER_BONUS[input.tier];
  const factor = Math.max(0.5, attemptFactor(attempts) - 0.15 * Math.max(0, input.hintsUsed));
  const streak = firstTry ? streakMultiplier(streakAfter) : 1;
  const clarity = Math.round(25 * Math.min(1, Math.max(0, input.targetScore)));
  const persistence = attempts >= 4 ? 20 : 0;
  const points = Math.round(base * factor * streak) + clarity + persistence;
  const lines: ScoreLine[] = [{ label: 'Memory restored', value: `+${base}` }];
  if (factor < 1) lines.push({ label: `${attempts} attempts${input.hintsUsed ? `, ${input.hintsUsed} echo${input.hintsUsed > 1 ? 'es' : ''}` : ''}`, value: `×${factor.toFixed(2)}` });
  if (streak > 1) lines.push({ label: `Streak of ${streakAfter}`, value: `×${streak.toFixed(1)}` });
  lines.push({ label: 'Signal clarity', value: `+${clarity}` });
  if (persistence) lines.push({ label: 'Didn’t give up', value: `+${persistence}` });
  return { points, fragments: firstTry ? 2 : 1, streakAfter, firstTry, lines };
}

export function zoneBonus(hintsUsedInZone: number): number {
  return 150 + (hintsUsedInZone === 0 ? 50 : 0);
}

export const RANKS = [
  { xp: 0, name: 'Apprentice signal tech' },
  { xp: 400, name: 'Memory technician' },
  { xp: 1100, name: 'Shape cartographer' },
  { xp: 2200, name: 'Recovery engineer' },
  { xp: 3600, name: 'Senior recovery engineer' },
  { xp: 5200, name: 'Keeper of the Atlas' },
];

export function rankFor(xp: number): { name: string; index: number; nextAt: number | null; progress: number } {
  let index = 0;
  for (let i = 0; i < RANKS.length; i++) if (xp >= RANKS[i].xp) index = i;
  const nextAt = RANKS[index + 1]?.xp ?? null;
  const progress = nextAt === null ? 1 : (xp - RANKS[index].xp) / (nextAt - RANKS[index].xp);
  return { name: RANKS[index].name, index, nextAt, progress };
}
