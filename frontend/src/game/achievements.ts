/** Achievements are checked against real progress after every recovery or zone completion. */
import type { Campaign } from './campaign';
import type { Progress } from './progress';

export interface Achievement {
  id: string;
  name: string;
  description: string;
  check: (p: Progress, c: Campaign, ctx: AchievementContext) => boolean;
}

export interface AchievementContext {
  /** The recovery that just happened, if any. */
  last?: { target: string; attempts: number; hintsUsed: number; clarity: number; previousMiss?: string };
}

const has = (p: Progress, names: string[]) => names.every(n => p.recovered[n]);

export const ACHIEVEMENTS: Achievement[] = [
  { id: 'first-light', name: 'First light', description: 'Restore your first memory.',
    check: p => Object.keys(p.recovered).length >= 1 },
  { id: 'clean-signal', name: 'Clean signal', description: 'Restore a memory on the first attempt, no echoes.',
    check: (_p, _c, x) => !!x.last && x.last.attempts === 1 && x.last.hintsUsed === 0 },
  { id: 'streak-3', name: 'In phase', description: 'Three first-try recoveries in a row.', check: p => p.bestStreak >= 3 },
  { id: 'streak-6', name: 'Resonance', description: 'Six first-try recoveries in a row.', check: p => p.bestStreak >= 6 },
  { id: 'stubborn', name: 'Stubborn engineer', description: 'Restore a memory after four or more attempts.',
    check: (_p, _c, x) => !!x.last && x.last.attempts >= 4 },
  { id: 'crystal', name: 'Crystal clear', description: 'Restore a memory with 99% signal clarity.',
    check: (_p, _c, x) => !!x.last && x.last.clarity >= 0.99 },
  { id: 'no-echoes', name: 'No echoes needed', description: 'Complete a zone without using an echo.',
    check: p => p.zonesCompleted.some(z => (p.hintsByZone[z] ?? 0) === 0) },
  { id: 'not-a-duck', name: 'Definitely not a duck', description: 'Restore “bird” after GLITCH mistook one of your birds for a duck.',
    check: p => !!p.recovered.bird && p.confusions.some(c => c.target === 'bird' && c.predicted === 'duck') },
  { id: 'crescent', name: 'Not a banana', description: 'Restore “moon” or “banana” after GLITCH confused the two.',
    check: p => p.confusions.some(c => (c.target === 'moon' && c.predicted === 'banana' && !!p.recovered.moon) || (c.target === 'banana' && c.predicted === 'moon' && !!p.recovered.banana)) },
  { id: 'menagerie', name: 'Menagerie', description: 'Restore every animal in the Living Archive.',
    check: (p, c) => { const z = c.zones.find(z => z.id === 'living-archive'); return !!z && z.classes.length > 0 && has(p, z.classes); } },
  { id: 'cartographer', name: 'Cartographer', description: 'Restore a memory in every zone.',
    check: (p, c) => c.zones.filter(z => !z.final).every(z => z.classes.some(n => p.recovered[n])) },
  { id: 'reboot', name: 'Core reboot', description: 'Finish the campaign.', check: p => p.campaignComplete },
  { id: 'atlas', name: 'The whole Atlas', description: 'Restore every memory GLITCH ever had.',
    check: (p, c) => c.classes.length > 0 && has(p, c.classes) },
];

export function evaluateAchievements(p: Progress, c: Campaign, ctx: AchievementContext = {}): string[] {
  return ACHIEVEMENTS.filter(a => !p.achievements.includes(a.id) && a.check(p, c, ctx)).map(a => a.id);
}

export const achievementById = (id: string) => ACHIEVEMENTS.find(a => a.id === id);
