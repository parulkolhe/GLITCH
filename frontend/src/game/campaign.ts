/**
 * Campaign structure: acts, memory zones and their target classes.
 *
 * Zones are authored against the 32 classes of the trained model, but the
 * campaign is always rebuilt from the classes the API actually reports
 * (/api/model). Unknown authored classes are dropped; model classes the
 * authoring does not mention are distributed into zones by difficulty tier,
 * so a retrained model with a different label set still yields a playable game.
 */

export type Tier = 'easy' | 'medium' | 'hard';

export interface ZoneSpec {
  id: string;
  act: 1 | 2 | 3;
  name: string;
  /** Short line under the zone name. */
  subtitle: string;
  classes: string[];
  /** Memories to recover before the zone's gate opens. */
  required: number;
  /** Visual identity: hue in degrees for the procedural world, plus grid style. */
  hue: number;
  grid: 'flat' | 'waves' | 'rings' | 'shards' | 'spiral' | 'core';
  final?: boolean;
}

export const ACTS: Record<1 | 2 | 3, { title: string; line: string }> = {
  1: { title: 'Act I: The Last Signal', line: 'Something arrived all at once. Then everything went quiet.' },
  2: { title: 'Act II: Fragments of Yesterday', line: 'Words without shapes. Shapes without names.' },
  3: { title: 'Act III: The Core Reboot', line: 'Everything you restored, all at once. Carefully this time.' },
};

export const AUTHORED_ZONES: ZoneSpec[] = [
  { id: 'boot-sector', act: 1, name: 'Boot Sector', subtitle: 'The first shapes GLITCH ever learned.',
    classes: ['house', 'sun', 'mountain', 'umbrella', 'star'], required: 3, hue: 78, grid: 'flat' },
  { id: 'lost-basics', act: 2, name: 'The Lost Basics', subtitle: 'Things people keep on tables and in pockets.',
    classes: ['apple', 'book', 'clock', 'chair', 'cake', 'key', 'banana', 'pizza'], required: 4, hue: 48, grid: 'waves' },
  { id: 'forgotten-world', act: 2, name: 'The Forgotten World', subtitle: 'Weather, water and things that grow.',
    classes: ['tree', 'flower', 'cloud', 'lightning', 'fish', 'butterfly', 'moon'], required: 4, hue: 165, grid: 'rings' },
  { id: 'signal-break', act: 2, name: 'The Signal Break', subtitle: 'Machines for going places and making noise.',
    classes: ['car', 'bicycle', 'airplane', 'sailboat', 'guitar', 'headphones'], required: 4, hue: 195, grid: 'shards' },
  { id: 'living-archive', act: 2, name: 'The Living Archive', subtitle: 'The animals. GLITCH mixes these up the most.',
    classes: ['cat', 'dog', 'bird', 'duck', 'frog', 'rabbit'], required: 3, hue: 265, grid: 'spiral' },
  { id: 'core-reboot', act: 3, name: 'Core Reboot', subtitle: 'Every zone at once. Hold the signal steady.',
    classes: [], required: 5, hue: 72, grid: 'core', final: true },
];

export const FINAL_SEQUENCE_LENGTH = 7;

export interface Campaign {
  zones: ZoneSpec[];
  classes: string[];
  tiers: Record<string, Tier>;
}

export function buildCampaign(modelClasses: string[], tiers: Record<string, string>): Campaign {
  const known = new Set(modelClasses);
  const normTiers: Record<string, Tier> = {};
  for (const name of modelClasses) {
    const t = tiers[name];
    normTiers[name] = t === 'easy' || t === 'hard' ? t : 'medium';
  }
  const zones = AUTHORED_ZONES.map(z => ({ ...z, classes: z.classes.filter(c => known.has(c)) }));
  const used = new Set(zones.flatMap(z => z.classes));
  const playable = zones.filter(z => !z.final);
  // Distribute unauthored classes: easy ones early, hard ones late.
  const orphans = modelClasses.filter(c => !used.has(c));
  for (const name of orphans) {
    const tier = normTiers[name];
    const pool = tier === 'easy' ? playable.slice(0, 2) : tier === 'hard' ? playable.slice(-2) : playable.slice(1, -1);
    const zone = (pool.length ? pool : playable).reduce((a, b) => (a.classes.length <= b.classes.length ? a : b));
    zone.classes.push(name);
  }
  for (const zone of zones) {
    if (!zone.final) zone.required = Math.max(1, Math.min(zone.required, zone.classes.length));
  }
  return { zones: zones.filter(z => z.final || z.classes.length > 0), classes: [...modelClasses], tiers: normTiers };
}

/** Deterministic PRNG (mulberry32) so tests and replays are reproducible. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const TIER_ORDER: Record<Tier, number> = { easy: 0, medium: 1, hard: 2 };

/**
 * Order of prompts inside a zone: easier first, shuffled within a tier.
 * Recovered classes are excluded; skipped classes go to the back of the queue.
 */
export function zoneQueue(zone: ZoneSpec, campaign: Campaign, recovered: Set<string>, skipped: string[], seed: number): string[] {
  const random = rng(seed);
  const open = zone.classes.filter(c => !recovered.has(c));
  const shuffled = open.map(c => ({ c, k: TIER_ORDER[campaign.tiers[c] ?? 'medium'] + random() * 0.9 }));
  shuffled.sort((a, b) => a.k - b.k);
  const fresh = shuffled.map(s => s.c).filter(c => !skipped.includes(c));
  const later = skipped.filter(c => open.includes(c));
  return [...fresh, ...later];
}

/**
 * Act III mixes earlier memories. It prefers classes the player skipped or
 * struggled with (a second chance), then the hardest recovered ones, then
 * anything never seen, and never repeats a class within the sequence.
 */
export function finalSequence(campaign: Campaign, history: {
  recovered: Record<string, { attempts: number }>; skipped: string[]; seen: string[];
}, seed: number, length = FINAL_SEQUENCE_LENGTH): string[] {
  const random = rng(seed);
  const shuffle = <T,>(xs: T[]) => xs.map(x => ({ x, k: random() })).sort((a, b) => a.k - b.k).map(o => o.x);
  const recovered = Object.keys(history.recovered);
  const struggled = recovered.filter(c => history.recovered[c].attempts >= 3);
  const skipped = history.skipped.filter(c => !history.recovered[c]);
  const hardRecovered = recovered.filter(c => campaign.tiers[c] === 'hard' && !struggled.includes(c));
  const seenRest = recovered.filter(c => !struggled.includes(c) && !hardRecovered.includes(c));
  const unseen = campaign.classes.filter(c => !history.seen.includes(c) && !history.recovered[c]);
  const ordered = [...shuffle(skipped), ...shuffle(struggled), ...shuffle(hardRecovered), ...shuffle(unseen), ...shuffle(seenRest)];
  const unique = [...new Set(ordered)];
  // Interleave so the sequence alternates difficulty instead of front-loading every hard class.
  const picked = unique.slice(0, length);
  const easyFirst = picked.filter(c => campaign.tiers[c] !== 'hard');
  const hard = picked.filter(c => campaign.tiers[c] === 'hard');
  const out: string[] = [];
  while (easyFirst.length || hard.length) {
    if (easyFirst.length) out.push(easyFirst.shift()!);
    if (hard.length) out.push(hard.shift()!);
  }
  return out;
}

export function zoneById(campaign: Campaign, id: string): ZoneSpec | undefined {
  return campaign.zones.find(z => z.id === id);
}

export function zoneOfClass(campaign: Campaign, name: string): ZoneSpec | undefined {
  return campaign.zones.find(z => !z.final && z.classes.includes(name));
}

export function articleFor(word: string): string {
  return /^[aeiou]/i.test(word) ? 'an' : 'a';
}
