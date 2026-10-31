/**
 * Everything GLITCH says. Written by hand; no language model is involved.
 *
 * Cinematic scripts are lists of beats. Reactions are chosen from the real
 * prediction: target, predicted class, calibrated score, attempt number,
 * campaign progress and known confusion pairs. A small memory of recent lines
 * keeps GLITCH from repeating itself.
 */
import type { Mood } from '../robot/moods';
import { articleFor } from '../game/campaign';

export type Speaker = 'glitch' | 'narrator' | 'log' | 'system';

export interface Beat {
  speaker: Speaker;
  text: string;
  mood?: Mood;
  /** Visual event the cinematic layer should play during this beat. */
  fx?: 'atlas' | 'atlas-draw' | 'signal' | 'overflow' | 'shatter' | 'silence' | 'boot' | 'flicker' | 'core' | 'converge' | 'reboot' | 'world' | 'log';
  choices?: { label: string; reply?: string }[];
}

// --------------------------------------------------------------- Act I intro
export const INTRO: Beat[] = [
  { speaker: 'narrator', fx: 'atlas', text: 'Before the signal, there was the Atlas.' },
  { speaker: 'narrator', fx: 'atlas-draw', text: 'A world drawn by everyone. Every house sketched on a napkin. Every lopsided sun in the corner of a page.' },
  { speaker: 'narrator', fx: 'atlas-draw', text: 'One small robot kept it in order. It looked at shapes, and it remembered what they were called.' },
  { speaker: 'narrator', fx: 'signal', text: 'Then, at 03:14 on an ordinary night, fifty million drawings arrived at once.' },
  { speaker: 'narrator', fx: 'overflow', text: 'It tried to remember every single one of them.' },
  { speaker: 'narrator', fx: 'shatter', text: '' },
  { speaker: 'narrator', fx: 'silence', text: 'The words survived. The shapes did not.' },
  { speaker: 'glitch', fx: 'boot', mood: 'booting', text: '—llo? Hello? Is that a person-shaped input?' },
  { speaker: 'glitch', fx: 'flicker', mood: 'confused', text: 'I’m GLITCH. I think. The name is still on file. Most other things are static.',
    choices: [{ label: 'What happened to you?', reply: 'I don’t know yet. There’s a log, but it’s scrambled. I remember a very loud night and then… this.' },
              { label: 'Nice to meet you, GLITCH.', reply: 'Nice to meet you too! Probably. My “nice” sensor is offline, but I’m choosing to believe it.' }] },
  { speaker: 'glitch', mood: 'thinking', text: 'Here’s the strange part. I can still read words. STAR. HOUSE. UMBRELLA. I just can’t remember what any of them look like.' },
  { speaker: 'glitch', mood: 'watching', text: 'You have hands. Hands make shapes. Would you draw for me?' },
];

// --------------------------------------------------------- core activation
export const CORE_ACTIVATION: Beat[] = [
  { speaker: 'glitch', mood: 'idle', text: 'First, my memory core. It’s offline. Press and hold it — gently, it’s the only one I have.' },
];

export const CORE_ONLINE: Beat[] = [
  { speaker: 'glitch', mood: 'recognising', fx: 'core', text: 'Oh. Oh, that’s warm. Recognition network: online. Memories restored: zero. Great start.' },
  { speaker: 'glitch', mood: 'idle', text: 'Here’s how this works. I give you a word. You draw it your way. No tracing — I don’t have anything to trace anyway.' },
  { speaker: 'glitch', mood: 'watching', text: 'If my network sees the same thing in your drawing, the memory comes back. I’m not grading art. I’m trying to recognise things.' },
  { speaker: 'system', text: 'GLITCH’s recognition network was trained once, long ago, on Quick, Draw! sketches. When you draw, it looks and guesses. It does not learn from you or change.' },
];

// ---------------------------------------------------------- zone intros
export const ZONE_INTROS: Record<string, Beat[]> = {
  'boot-sector': [
    { speaker: 'glitch', mood: 'idle', text: 'Boot Sector. The first shapes I ever learned. Houses. Suns. Probably very simple. Probably.' },
  ],
  'lost-basics': [
    { speaker: 'glitch', mood: 'excited', text: 'I remember the sun now! Big circle, lines poking out, extremely hot. I feel four percent more like myself.' },
    { speaker: 'glitch', mood: 'thinking', text: 'Next: the Lost Basics. Things people keep on tables. There’s a word in here called “pizza” and frankly it’s haunting me.' },
  ],
  'forgotten-world': [
    { speaker: 'log', fx: 'log', text: 'LOG 03:14:09 — 50,000,000 drawings queued. Instruction: remember. Reply: all of them?' },
    { speaker: 'glitch', mood: 'confused', text: 'That’s… my log. From the night of the signal. I asked a question and nobody answered.' },
    { speaker: 'glitch', mood: 'idle', text: 'Let’s go outside. The Forgotten World: weather, water, things that grow. I miss weather.' },
  ],
  'signal-break': [
    { speaker: 'log', fx: 'log', text: 'LOG 03:14:10 — Storage at 340%. Keeping only the newest drawings. Losing the old ones. Losing the—' },
    { speaker: 'glitch', mood: 'thinking', text: 'The drawings came from everywhere. Kids. Grandparents. Someone who drew four hundred bicycles in a row. I respect them.' },
    { speaker: 'glitch', mood: 'excited', text: 'The Signal Break: machines for going places and making noise. My favourite kind of noise is the kind I can recognise.' },
  ],
  'living-archive': [
    { speaker: 'glitch', mood: 'frustrated', text: 'The animals. Fair warning: I was always bad at these.' },
    { speaker: 'glitch', mood: 'confused', text: 'Birds and ducks? Same shape, different attitude. Cats and dogs? Don’t get me started. My network mixes these up more than anything else.' },
    { speaker: 'glitch', mood: 'watching', text: 'If I get it wrong, it’s my memory, not your drawing. Give me the one detail only that animal has.' },
  ],
  'core-reboot': [
    { speaker: 'log', fx: 'log', text: 'LOG 03:14:11 — Emergency protocol: separate words from shapes. If the shapes are lost, the words survive. Someone will come.' },
    { speaker: 'glitch', mood: 'thinking', text: 'I know what the signal was now. It wasn’t an attack. Nobody broke me.' },
    { speaker: 'glitch', mood: 'confused', text: 'It was everyone’s drawings, all at once, and I tried to keep a perfect copy of every one. My core overflowed and cut the words loose to save itself.' },
    { speaker: 'glitch', mood: 'recovering', text: 'But you never drew me a perfect copy of anything. You drew your own versions, and I still knew them.' },
    { speaker: 'glitch', mood: 'excited', text: 'I don’t need every drawing. I need the idea inside them. Let’s reboot the core. Everything, together.' },
  ],
};

export const ZONE_LOGS: Record<string, string> = {
  'boot-sector': 'LOG 03:14:07 — Incoming transmission. Size: unusual. Origin: everywhere.',
  'lost-basics': 'LOG 03:14:08 — Opening the transmission. It is full of drawings. Mostly cats.',
  'forgotten-world': 'LOG 03:14:10 — Every drawing is slightly different. I keep trying to store all the differences.',
  'signal-break': 'LOG 03:14:11 — The words are safe. The shapes are going dark one by one.',
  'living-archive': 'LOG 03:14:12 — Note for whoever comes: they don’t have to be good at drawing. Tell them that first.',
  'core-reboot': 'LOG 00:00:00 — Reboot complete. Somebody came.',
};

// ------------------------------------------------------------------ finale
export interface FinaleFacts {
  recovered: number;
  total: number;
  skipped: string[];
  bestStreak: number;
  topConfusion: { target: string; predicted: string; count: number } | null;
  clearest: { name: string; clarity: number } | null;
  hardestWon: { name: string; attempts: number } | null;
  attempts: number;
}

export function finaleBeats(f: FinaleFacts): Beat[] {
  const beats: Beat[] = [
    { speaker: 'glitch', mood: 'recovering', fx: 'converge', text: 'Bringing everything you drew into the core. Hold on, this might tickle.' },
    { speaker: 'glitch', mood: 'scanning', fx: 'reboot', text: 'Rebooting. Words, meet shapes. Shapes, meet words. Please get along.' },
    { speaker: 'glitch', mood: 'restored', fx: 'world', text: `Look. The Atlas is back, and it’s made of your drawings. ${f.recovered} of ${f.total} memories, in your handwriting.` },
  ];
  if (f.recovered < f.total) {
    beats.push({ speaker: 'glitch', mood: 'idle', fx: 'world', text: `${f.total - f.recovered} are still dark. Dark stars are still stars. We can go back for them any time from the archive.` });
  } else {
    beats.push({ speaker: 'glitch', mood: 'excited', fx: 'world', text: 'Every single one. The whole Atlas. I didn’t think that was possible. I’m going to be insufferable about this.' });
  }
  if (f.hardestWon && f.hardestWon.attempts >= 3) {
    beats.push({ speaker: 'glitch', mood: 'restored', fx: 'world', text: `The ${f.hardestWon.name} took ${f.hardestWon.attempts} tries. You didn’t give up on it. I won’t forget that one.` });
  }
  if (f.topConfusion) {
    const { target, predicted } = f.topConfusion;
    beats.push({ speaker: 'glitch', mood: 'thinking', fx: 'world', text: `For the record, I still think one of your ${target}s looked a bit like ${articleFor(predicted)} ${predicted}. My network does that. Don’t tell the ${target}.` });
  } else {
    beats.push({ speaker: 'glitch', mood: 'thinking', fx: 'world', text: 'You never once confused me. That’s either talent or a very lucky night.' });
  }
  beats.push(
    { speaker: 'glitch', mood: 'idle', fx: 'world', text: 'One honest thing: you didn’t change my network. It was trained before you got here. What you did was harder. You found the shapes it already knew.' },
    { speaker: 'glitch', mood: 'restored', fx: 'world', text: 'Memory Recovery Protocol: complete. Thank you, engineer. I remember you, too.' },
  );
  return beats;
}

// --------------------------------------------------------------- reactions
export interface ReactionContext {
  target: string;
  predicted: string;
  second?: string;
  score: number;         // calibrated top-1 score
  targetScore: number;   // calibrated score of the target class
  targetRank: number | null;
  attempt: number;       // 1-based
  uncertain: boolean;
  streak: number;
  integrity: number;     // 0..1 campaign progress
}

const recent: string[] = [];
function pick(options: string[], seed: number): string {
  const fresh = options.filter(o => !recent.includes(o));
  const pool = fresh.length ? fresh : options;
  const line = pool[Math.abs(Math.floor(seed)) % pool.length];
  recent.push(line);
  if (recent.length > 12) recent.shift();
  return line;
}
const pct = (x: number) => `${Math.round(x * 100)}%`;
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
const fill = (s: string, c: ReactionContext) => cap(
  s.replaceAll('{target}', c.target).replaceAll('{pred}', c.predicted).replaceAll('{second}', c.second ?? 'something else')
    .replaceAll('{a_pred}', `${articleFor(c.predicted)} ${c.predicted}`).replaceAll('{a_target}', `${articleFor(c.target)} ${c.target}`)
    .replaceAll('{pct}', pct(c.score)).replaceAll('{streak}', String(c.streak)));

/** Lines for the model's real, frequent confusions (from the test confusion matrix). */
const PAIR_LINES: Record<string, string[]> = {
  'bird>duck': ['A duck. I see a duck. I know it’s supposed to be a bird. Ducks are just birds with paperwork. Try a pointed beak, no water.'],
  'duck>bird': ['That’s a bird to me. Ducks need a pond and a flat bill, otherwise they’re just birds on holiday.'],
  'cat>dog': ['Woof? I’m getting dog. Cats have pointy triangle ears and whiskers. Dogs have the floppy kind.'],
  'dog>cat': ['That’s a cat. A very cat cat. Dogs usually have a longer snout and ears that hang down.'],
  'moon>banana': ['Banana! Wait. No. Is it? The moon is a thinner crescent with sharp tips. Bananas get a stalk.'],
  'banana>moon': ['That’s a moon. A lovely moon. Bananas are chunkier, with a little stalk on one end.'],
  'dog>frog': ['A frog? Something about the face is very frog. Dogs have a longer snout and a tail.'],
  'frog>dog': ['I’m seeing a dog. Frogs have big eyes on top of a wide, flat head and bent legs.'],
  'rabbit>cat': ['Cat, says my network. Rabbits need those long upright ears. Really long. Longer than that.'],
  'cat>rabbit': ['A rabbit! Shorter ears would make it a cat. Triangles, not tall ovals.'],
  'sun>flower': ['A flower. A sunny flower. The sun is usually just a circle and straight rays, no stem.'],
  'flower>sun': ['That’s a sun to me. A stem and leaves would turn it into a flower.'],
  'airplane>bird': ['A bird? Planes have a long straight body and stiff wings. Birds are allowed to be wobbly.'],
  'bird>airplane': ['An airplane! Birds are rounder, with a head and a beak, not a fuselage.'],
  'clock>apple': ['An apple? Clocks need hands pointing out from the centre.'],
  'key>guitar': ['That’s a guitar. Keys are smaller: a ring at one end and teeth at the other.'],
  'mountain>house': ['A house? Mountains don’t have doors. Just peaks.'],
  'house>mountain': ['A mountain. Add walls and a door so I know someone lives there.'],
};

export function recognizedLine(c: ReactionContext, seed: number): string {
  if (c.attempt >= 4) {
    return fill(pick([
      'THERE it is. {a_target}. You kept going. I’d hug you but my arms are mostly decorative.',
      '{a_target}! Attempt {attempt}, and worth every one. That memory was stuck deep.',
    ], seed), c).replace('{attempt}', String(c.attempt));
  }
  if (c.streak >= 3) {
    return fill(pick([
      'That’s {streak} in a row. My core is humming. Is this what confidence feels like?',
      '{streak} first-try memories. I’m starting to remember how to remember.',
    ], seed), c);
  }
  if (c.targetScore >= 0.97) {
    return fill(pick([
      'No doubt at all. That is {a_target}. I’d know it anywhere. Well, now I would.',
      '{a_target}. Crisp. Clear. I remember {target}s.',
      'Instant match. {a_target}! Where have you been all my life?',
    ], seed), c);
  }
  if (c.targetScore < 0.6) {
    return fill(pick([
      'It’s {a_target}… yes. Mostly yes. Memory restored, slightly wobbly.',
      '{a_target}, my network says, though it hesitated a little. I’ll take it.',
    ], seed), c);
  }
  if (c.integrity < 0.2) {
    return fill(pick([
      'Oh! {a_target}! I remember {target}s. I remember remembering them.',
      'That’s {a_target}. Something just clicked back into place in here.',
    ], seed), c);
  }
  return fill(pick([
    'That’s {a_target}. Memory restored.',
    '{a_target}! Another piece of the Atlas lights up.',
    'Recognised. {a_target}, filed where it belongs.',
  ], seed), c);
}

export function missedLine(c: ReactionContext, seed: number): string {
  if (c.uncertain) {
    return fill(pick([
      'I… don’t know. My best guess is {a_pred}, but only {pct}. Try making the main shape bigger and bolder.',
      'Static. It’s somewhere between {pred} and {second}, and I won’t pretend I’m sure.',
      'Nothing clicks yet. Give it the one feature only {a_target} has.',
    ], seed), c);
  }
  const pair = PAIR_LINES[`${c.target}>${c.predicted}`];
  if (pair && c.attempt <= 3) return fill(pick(pair, seed), c);
  if (c.attempt >= 3) {
    return fill(pick([
      'Still seeing {a_pred}. That’s on my scrambled memory, not you. An echo might help.',
      '{a_pred} again. My {target} memory is badly tangled. Clear it and try a different shape, or ask for an echo.',
    ], seed), c);
  }
  if (c.targetRank !== null && c.targetRank <= 3) {
    return fill(pick([
      'So close. I’m mostly seeing {a_pred}, but {target} is right behind it. Exaggerate what makes it {a_target}.',
      'Wait, that looks familiar… {pred} first, {target} a close runner-up. One more clear detail.',
    ], seed), c);
  }
  return fill(pick([
    '{a_pred}? I’m getting {pred} from this. Which is lovely, just not {a_target}.',
    'Hmm. My network says {pred}, quite firmly. Want to try a different take on {a_target}?',
    '{a_pred}! Is it {a_pred}? It’s {a_pred}, isn’t it. No? Different angle, then.',
  ], seed), c);
}

export function errorLine(kind: string): string {
  switch (kind) {
    case 'timeout': return 'My recognition core went quiet for too long. Your drawing is still here. Send it again?';
    case 'unavailable': return 'My recognition core isn’t loaded on the server. Your drawing is safe; check the backend log, then try again.';
    case 'invalid-input': return 'I couldn’t read that drawing. Make sure there are some visible strokes, then send again.';
    default: return 'I’ve lost the link to my recognition core. Your drawing is safe. Check that the backend is running, then send it again.';
  }
}

export const READY_LINES = [
  'Ready when you are. Draw {a_target}.',
  'Next word: {target}. Your version, please.',
  'I have the word “{target}” and nothing else. Help?',
  '{target}. I can almost picture it. Almost.',
];

export function readyLine(target: string, seed: number): string {
  return cap(pick(READY_LINES, seed).replaceAll('{a_target}', `${articleFor(target)} ${target}`).replaceAll('{target}', target));
}
