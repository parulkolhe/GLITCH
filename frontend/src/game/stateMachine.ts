/**
 * The game's explicit phase machine. Every screen is a phase; every change of
 * phase is an event with a row in TRANSITIONS. Anything not in the table is
 * rejected, so impossible combinations (a prediction resolving after the
 * player cleared the canvas, a restoration starting twice) cannot happen.
 */

export const PHASES = [
  'BOOT', 'TITLE', 'INTRO_CINEMATIC', 'MEMORY_CORE_ACTIVATION', 'ZONE_INTRO',
  'DRAWING_READY', 'DRAWING_IN_PROGRESS', 'PREDICTION_PENDING',
  'PREDICTION_RECOGNIZED', 'PREDICTION_UNCERTAIN', 'PREDICTION_MISSED', 'PREDICTION_ERROR',
  'MEMORY_RESTORATION', 'LEVEL_COMPLETE', 'FINAL_SEQUENCE', 'CAMPAIGN_COMPLETE',
] as const;
export type Phase = typeof PHASES[number];

export type PhaseEvent =
  | 'BOOT_DONE' | 'NEW_GAME' | 'CONTINUE' | 'INTRO_DONE' | 'CORE_ONLINE' | 'ZONE_READY'
  | 'STROKE_START' | 'CLEAR' | 'SUBMIT'
  | 'RESULT_RECOGNIZED' | 'RESULT_UNCERTAIN' | 'RESULT_MISSED' | 'RESULT_ERROR'
  | 'RESTORE' | 'NEXT_TARGET' | 'ZONE_DONE' | 'SKIP_TARGET'
  | 'NEXT_ZONE' | 'BEGIN_FINALE' | 'FINALE_DONE' | 'TO_TITLE' | 'PRACTICE';

const RETRYABLE: Phase[] = ['PREDICTION_MISSED', 'PREDICTION_UNCERTAIN', 'PREDICTION_ERROR'];
const DRAWING: Phase[] = ['DRAWING_READY', 'DRAWING_IN_PROGRESS', ...RETRYABLE];

type Table = Partial<Record<Phase, Partial<Record<PhaseEvent, Phase>>>>;

function build(): Table {
  const t: Table = {
    BOOT: { BOOT_DONE: 'TITLE' },
    TITLE: { NEW_GAME: 'INTRO_CINEMATIC', CONTINUE: 'ZONE_INTRO', PRACTICE: 'DRAWING_READY' },
    INTRO_CINEMATIC: { INTRO_DONE: 'MEMORY_CORE_ACTIVATION' },
    MEMORY_CORE_ACTIVATION: { CORE_ONLINE: 'ZONE_INTRO' },
    ZONE_INTRO: { ZONE_READY: 'DRAWING_READY' },
    DRAWING_READY: { STROKE_START: 'DRAWING_IN_PROGRESS', SKIP_TARGET: 'DRAWING_READY', ZONE_DONE: 'LEVEL_COMPLETE' },
    DRAWING_IN_PROGRESS: { STROKE_START: 'DRAWING_IN_PROGRESS', CLEAR: 'DRAWING_READY', SUBMIT: 'PREDICTION_PENDING', SKIP_TARGET: 'DRAWING_READY' },
    PREDICTION_PENDING: {
      RESULT_RECOGNIZED: 'PREDICTION_RECOGNIZED', RESULT_UNCERTAIN: 'PREDICTION_UNCERTAIN',
      RESULT_MISSED: 'PREDICTION_MISSED', RESULT_ERROR: 'PREDICTION_ERROR',
      CLEAR: 'DRAWING_READY',  // clearing cancels the request; its late answer is discarded
    },
    PREDICTION_RECOGNIZED: { RESTORE: 'MEMORY_RESTORATION' },
    MEMORY_RESTORATION: { NEXT_TARGET: 'DRAWING_READY', ZONE_DONE: 'LEVEL_COMPLETE' },
    LEVEL_COMPLETE: { NEXT_ZONE: 'ZONE_INTRO', BEGIN_FINALE: 'FINAL_SEQUENCE' },
    FINAL_SEQUENCE: { FINALE_DONE: 'CAMPAIGN_COMPLETE' },
    CAMPAIGN_COMPLETE: {},
  };
  for (const phase of RETRYABLE) {
    t[phase] = { STROKE_START: 'DRAWING_IN_PROGRESS', CLEAR: 'DRAWING_READY', SUBMIT: 'PREDICTION_PENDING', SKIP_TARGET: 'DRAWING_READY' };
  }
  // Returning to the title is always allowed (pause menu), except mid-boot.
  for (const phase of PHASES) {
    if (phase === 'BOOT' || phase === 'TITLE') continue;
    t[phase] = { ...(t[phase] || {}), TO_TITLE: 'TITLE' };
  }
  return t;
}

export const TRANSITIONS: Readonly<Table> = build();

export function next(phase: Phase, event: PhaseEvent): Phase | null {
  return TRANSITIONS[phase]?.[event] ?? null;
}

export function canSend(phase: Phase, event: PhaseEvent): boolean {
  return next(phase, event) !== null;
}

export const isDrawingPhase = (phase: Phase) => DRAWING.includes(phase);
export const isRetryPhase = (phase: Phase) => RETRYABLE.includes(phase);

export interface Machine {
  phase: Phase;
  /** Monotonic counter; changes on every accepted transition. */
  serial: number;
  lastEvent: PhaseEvent | null;
  rejected: number;
}

export const initialMachine: Machine = { phase: 'BOOT', serial: 0, lastEvent: null, rejected: 0 };

export function reduce(machine: Machine, event: PhaseEvent): Machine {
  const target = next(machine.phase, event);
  if (target === null) {
    if (typeof console !== 'undefined' && (globalThis as any).__GLITCH_DEBUG__) {
      console.warn(`[phase] rejected ${event} in ${machine.phase}`);
    }
    return { ...machine, rejected: machine.rejected + 1 };
  }
  return { phase: target, serial: machine.serial + 1, lastEvent: event, rejected: machine.rejected };
}
