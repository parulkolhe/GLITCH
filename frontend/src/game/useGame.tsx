/**
 * Game controller. Owns the phase machine, the campaign progress, the current
 * drawing session and the prediction requests, and emits every gameplay event.
 * Screens read state from here and call its actions; they never talk to the
 * API or to each other directly.
 *
 * Success rule (the whole game hinges on it): a memory is restored only when
 * the CNN's top-1 class equals the target. Confidence shapes feedback, hints
 * and the clarity bonus, but can never turn a wrong class into a success.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useReducer, useRef, useState, type ReactNode } from 'react';
import { reduce, initialMachine, type Machine, type Phase, type PhaseEvent, canSend } from './stateMachine';
import { AUTHORED_ZONES, buildCampaign, finalSequence, zoneQueue, type Campaign, type ZoneSpec, type Tier } from './campaign';
import { scoreRecovery, zoneBonus, type ScoreLine } from './scoring';
import {
  completeZone, integrity as integrityOf, markSeen, newProgress, recordHint, recordMiss, recordRecovery, recordSkip,
  zoneRecoveredCount, type Progress,
} from './progress';
import { loadProgress, saveProgress, clearProgress, loadSettings, saveSettings, type Settings } from './save';
import { evaluateAchievements } from './achievements';
import { PredictionClient, PredictionError, type ModelInfo, type PredictionResult } from '../services/predictionClient';
import { compactStrokes, exportPng, inkAmount, type Stroke } from '../canvas/strokes';
import { bus } from '../events/bus';
import { errorLine, missedLine, readyLine, recognizedLine } from '../story/dialogue';
import type { Mood } from '../robot/moods';

export type Mode = 'campaign' | 'practice' | 'rush' | 'revisit';
export const RUSH_SECONDS = 75;

export interface Session {
  target: string | null;
  zoneId: string | null;
  attempts: number;
  hintLevel: number;
  result: PredictionResult | null;
  error: { kind: string; message: string } | null;
  submittedInk: number | null;   // ink amount at last submission, to block identical resubmits
  rushEndsAt: number | null;
  rushRecovered: string[];
  rushScore: number;
  revisitQueue: string[];
  previousMiss: string | null;
}

export interface Restoration {
  target: string;
  zoneId: string | null;
  points: number;
  lines: ScoreLine[];
  firstTry: boolean;
  streak: number;
  clarity: number;
  achievements: string[];
  zoneDone: boolean;
  strokes: number[][];
}

const emptySession = (): Session => ({
  target: null, zoneId: null, attempts: 0, hintLevel: 0, result: null, error: null, submittedInk: null,
  rushEndsAt: null, rushRecovered: [], rushScore: 0, revisitQueue: [], previousMiss: null,
});

function systemReducedMotion(): boolean {
  try { return window.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch { return false; }
}

export function useGameState() {
  const [machine, dispatch] = useReducer(reduce, initialMachine);
  const machineRef = useRef<Machine>(machine);
  machineRef.current = machine;
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [progress, setProgress] = useState<Progress>(() => loadProgress() ?? newProgress());
  const [hasSave, setHasSave] = useState<boolean>(() => !!loadProgress());
  const progressRef = useRef(progress);
  progressRef.current = progress;
  const [settings, setSettingsState] = useState<Settings>(() => loadSettings());
  const [mode, setMode] = useState<Mode>('campaign');
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const [session, setSession] = useState<Session>(emptySession);
  const sessionRef = useRef(session);
  sessionRef.current = session;
  const [line, setLine] = useState<{ text: string; key: number; mood?: Mood } | null>(null);
  const [restoration, setRestoration] = useState<Restoration | null>(null);
  const [toast, setToast] = useState<{ id: string; key: number } | null>(null);
  const client = useMemo(() => new PredictionClient(''), []);
  const lineKey = useRef(0);

  const send = useCallback((event: PhaseEvent) => {
    const before = machineRef.current;
    const after = reduce(before, event);
    if ((globalThis as any).__GLITCH_DEBUG__) console.info(`[phase] ${before.phase} --${event}--> ${after.serial !== before.serial ? after.phase : 'REJECTED'}`);
    if (after.serial !== before.serial) { machineRef.current = after; dispatch(event); return true; }
    return false;
  }, []);

  const say = useCallback((text: string, mood?: Mood) => { lineKey.current += 1; setLine({ text, key: lineKey.current, mood }); }, []);

  // --------------------------------------------------------------- model
  const loadModel = useCallback(async () => {
    setModelError(null);
    try {
      const info = await client.modelInfo();
      if (!info.modelReady) throw new PredictionError('unavailable', 'The server is running, but the model failed to load. See the backend log.');
      setModel(info);
      return info;
    } catch (error) {
      setModelError((error as Error).message);
      return null;
    }
  }, [client]);

  const campaign: Campaign = useMemo(() => {
    if (model && model.classes.length) return buildCampaign(model.classes, model.difficulty);
    const authored = AUTHORED_ZONES.flatMap(z => z.classes);
    return buildCampaign(authored, {});
  }, [model]);
  const campaignRef = useRef(campaign);
  campaignRef.current = campaign;

  const currentZone: ZoneSpec = campaign.zones[Math.min(progress.zoneIndex, campaign.zones.length - 1)];
  const integrity = integrityOf(progress, campaign);
  const reducedMotion = settings.reducedMotion === 'on' || (settings.reducedMotion === 'system' && systemReducedMotion());

  // ------------------------------------------------------------ persistence
  useEffect(() => {
    if (mode === 'practice' || mode === 'rush') return;
    const id = setTimeout(() => { if (saveProgress(progress)) setHasSave(true); }, 250);
    return () => clearTimeout(id);
  }, [progress, mode]);

  const updateSettings = useCallback((patch: Partial<Settings>) => {
    setSettingsState(prev => { const next = { ...prev, ...patch }; saveSettings(next); return next; });
    bus.emit('settings:changed', {});
  }, []);

  // ------------------------------------------------------------ targets
  const pickTarget = useCallback((p: Progress, zone: ZoneSpec | undefined, avoid: string | null): string | null => {
    const c = campaignRef.current;
    const m = modeRef.current;
    if (m === 'practice' || m === 'rush') {
      const pool = c.classes.filter(n => n !== avoid);
      return pool[Math.floor(Math.random() * pool.length)] ?? null;
    }
    if (m === 'revisit') {
      const q = sessionRef.current.revisitQueue.filter(n => !p.recovered[n]);
      return q.find(n => n !== avoid) ?? q[0] ?? null;
    }
    if (!zone) return null;
    if (zone.final) {
      const q = (p.finalQueue ?? []).filter(n => !p.finalRecovered.includes(n));
      const skippedLater = q.filter(n => p.skipped.includes(n));
      const fresh = q.filter(n => !p.skipped.includes(n));
      const ordered = [...fresh, ...skippedLater];
      return ordered.find(n => n !== avoid) ?? ordered[0] ?? null;
    }
    const q = zoneQueue(zone, c, new Set(Object.keys(p.recovered)), p.skipped, p.seed + p.zoneIndex);
    return q.find(n => n !== avoid) ?? q[0] ?? null;
  }, []);

  const zoneIsDone = (p: Progress, zone: ZoneSpec) =>
    zone.final ? p.finalRecovered.length >= zone.required : zoneRecoveredCount(p, zone.classes) >= zone.required;

  const beginTarget = useCallback((p: Progress, zone: ZoneSpec | undefined, avoid: string | null) => {
    const target = pickTarget(p, zone, avoid);
    setSession(s => ({ ...emptySession(), rushEndsAt: s.rushEndsAt, rushRecovered: s.rushRecovered, rushScore: s.rushScore,
      revisitQueue: s.revisitQueue, target, zoneId: zone?.id ?? null }));
    if (target) {
      setProgress(prev => markSeen(prev, target));
      say(readyLine(target, Math.random() * 1000), 'idle');
    }
    return target;
  }, [pickTarget, say]);

  // Act III's sequence is fixed the first time the final zone is entered.
  const ensureFinalQueue = useCallback((p: Progress): Progress => {
    if (p.finalQueue && p.finalQueue.length) return p;
    const c = campaignRef.current;
    const fin = c.zones.find(z => z.final);
    const queue = finalSequence(c, { recovered: p.recovered, skipped: p.skipped, seen: p.seen }, p.seed, Math.max(fin?.required ?? 5, 7));
    return { ...p, finalQueue: queue };
  }, []);

  // ------------------------------------------------------------- actions
  const startNewCampaign = useCallback(() => {
    clearProgress();
    const fresh = newProgress();
    setProgress(fresh);
    setMode('campaign');
    setSession(emptySession());
    send('NEW_GAME');
  }, [send]);

  const continueCampaign = useCallback(() => {
    setMode('campaign');
    setSession(emptySession());
    send('CONTINUE');
  }, [send]);

  const startPractice = useCallback((kind: 'practice' | 'rush') => {
    setMode(kind);
    modeRef.current = kind;
    const base = { ...emptySession(), rushEndsAt: kind === 'rush' ? Date.now() + RUSH_SECONDS * 1000 : null };
    setSession(base);
    sessionRef.current = base;
    if (send('PRACTICE')) beginTarget(progressRef.current, undefined, null);
  }, [send, beginTarget]);

  const startRevisit = useCallback((classes: string[]) => {
    const open = classes.filter(c => !progressRef.current.recovered[c]);
    if (!open.length) return;
    setMode('revisit');
    modeRef.current = 'revisit';
    const base = { ...emptySession(), revisitQueue: open };
    setSession(base);
    sessionRef.current = base;
    if (send('PRACTICE')) beginTarget(progressRef.current, undefined, null);
  }, [send, beginTarget]);

  const markIntroSeen = useCallback(() => setProgress(p => ({ ...p, introSeen: true })), []);

  const enterZone = useCallback(() => {
    const p0 = progressRef.current;
    const zone = campaignRef.current.zones[Math.min(p0.zoneIndex, campaignRef.current.zones.length - 1)];
    const p = zone.final ? ensureFinalQueue(p0) : p0;
    if (p !== p0) { setProgress(p); progressRef.current = p; }
    bus.emit('zone:enter', { zoneId: zone.id, act: zone.act });
    if (send('ZONE_READY')) beginTarget(p, zone, null);
  }, [send, beginTarget, ensureFinalQueue]);

  const strokeStarted = useCallback(() => {
    const phase = machineRef.current.phase;
    if (phase === 'PREDICTION_PENDING') return; // the canvas is locked while scanning
    send('STROKE_START');
  }, [send]);

  const clearDrawing = useCallback(() => {
    if (machineRef.current.phase === 'PREDICTION_PENDING') client.cancel();
    if (send('CLEAR')) {
      setSession(s => ({ ...s, result: null, error: null, submittedInk: null }));
      bus.emit('draw:clear', {});
    }
  }, [send, client]);

  const submitDrawing = useCallback(async (strokes: Stroke[]) => {
    const s = sessionRef.current;
    const ink = inkAmount(strokes);
    if (!s.target || ink < 8) { say('I need a few more lines than that. Draw the whole thing.', 'confused'); return; }
    if (s.submittedInk !== null && Math.abs(s.submittedInk - ink) < 0.5 && machineRef.current.phase !== 'PREDICTION_ERROR') {
      say('That’s the same drawing I just saw. Change something, or clear it and try a new take.', 'thinking');
      return;
    }
    if (!send('SUBMIT')) return;
    const target = s.target;
    const attempt = s.attempts + 1;
    const serial = machineRef.current.serial;
    setSession(prev => ({ ...prev, attempts: attempt, submittedInk: ink, result: null, error: null }));
    bus.emit('predict:start', { target });
    let png: string;
    try { png = exportPng(strokes); } catch (e) {
      send('RESULT_ERROR');
      setSession(prev => ({ ...prev, error: { kind: 'client', message: (e as Error).message } }));
      return;
    }
    const minScan = new Promise(r => setTimeout(r, 650)); // let the scan read as an event, not a flicker
    try {
      const [result] = await Promise.all([client.predict(png, target), minScan]);
      // Stale guard: the player cleared, skipped or left while we waited.
      if (machineRef.current.serial !== serial || machineRef.current.phase !== 'PREDICTION_PENDING' || sessionRef.current.target !== target) return;
      bus.emit('predict:result', { result, target, attempt });
      const top = result.predictions[0];
      const p = progressRef.current;
      const ctx = {
        target, predicted: top.className, second: result.predictions[1]?.className, score: top.score,
        targetScore: result.targetScore ?? 0, targetRank: result.targetRank, attempt, uncertain: result.uncertain,
        streak: p.streak, integrity: integrityOf(p, campaignRef.current),
      };
      setSession(prev => ({ ...prev, result }));
      if (top.className === target) {
        send('RESULT_RECOGNIZED');
        handleRecognized(result, strokes, attempt, ctx);
      } else {
        send(result.uncertain ? 'RESULT_UNCERTAIN' : 'RESULT_MISSED');
        if (modeRef.current !== 'practice' && modeRef.current !== 'rush') setProgress(prev => recordMiss(prev, target, top.className, result.uncertain));
        setSession(prev => ({ ...prev, previousMiss: top.className }));
        bus.emit('memory:missed', { target, predicted: top.className, uncertain: result.uncertain, attempt });
        say(missedLine(ctx, Math.random() * 1000), result.uncertain ? 'thinking' : attempt >= 3 ? 'frustrated' : 'confused');
      }
    } catch (error) {
      const err = error as PredictionError;
      if (err.kind === 'aborted') return;
      if (machineRef.current.serial !== serial) return;
      send('RESULT_ERROR');
      setSession(prev => ({ ...prev, error: { kind: err.kind ?? 'network', message: err.message } }));
      bus.emit('predict:error', { message: err.message });
      say(errorLine(err.kind), 'frustrated');
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [send, client, say]);

  const handleRecognized = (result: PredictionResult, strokes: Stroke[], attempt: number, ctx: Parameters<typeof recognizedLine>[0]) => {
    const p = progressRef.current;
    const c = campaignRef.current;
    const s = sessionRef.current;
    const target = s.target!;
    const m = modeRef.current;
    const tier: Tier = c.tiers[target] ?? 'medium';
    const clarity = result.targetScore ?? result.predictions[0].score;
    const score = scoreRecovery({ attempts: attempt, hintsUsed: s.hintLevel, streakBefore: m === 'rush' || m === 'practice' ? 0 : p.streak, tier, targetScore: clarity });
    const compact = compactStrokes(strokes);
    say(recognizedLine({ ...ctx, streak: score.streakAfter }, Math.random() * 1000), 'recognising');
    let achievements: string[] = [];
    let zoneDone = false;
    if (m === 'campaign' || m === 'revisit') {
      const zone = c.zones[Math.min(p.zoneIndex, c.zones.length - 1)];
      const isFinal = m === 'campaign' && !!zone.final;
      let next = recordRecovery(p, target, {
        zoneId: s.zoneId ?? '', attempts: attempt, hintsUsed: s.hintLevel, clarity, points: score.points,
        strokes: compact, at: Date.now(), modelVersion: result.modelVersion,
      }, score.streakAfter, score.fragments, isFinal);
      zoneDone = m === 'campaign' && zoneIsDone(next, zone);
      achievements = evaluateAchievements(next, c, { last: { target, attempts: attempt, hintsUsed: s.hintLevel, clarity } });
      if (achievements.length) next = { ...next, achievements: [...next.achievements, ...achievements] };
      setProgress(next);
      progressRef.current = next;
      if (m === 'revisit') zoneDone = false;
    } else if (m === 'rush') {
      setSession(prev => ({ ...prev, rushRecovered: [...prev.rushRecovered, target], rushScore: prev.rushScore + score.points }));
    }
    bus.emit('memory:recognized', { target, streak: score.streakAfter, points: score.points, firstTry: score.firstTry });
    if (score.streakAfter > 0 && score.streakAfter % 3 === 0) bus.emit('streak:milestone', { streak: score.streakAfter });
    achievements.forEach((id, i) => setTimeout(() => { bus.emit('achievement:unlock', { id }); setToast({ id, key: Date.now() + i }); }, 1400 + i * 2600));
    setRestoration({ target, zoneId: s.zoneId, points: score.points, lines: score.lines, firstTry: score.firstTry, streak: score.streakAfter,
      clarity, achievements, zoneDone, strokes: compact });
    setTimeout(() => send('RESTORE'), m === 'rush' ? 350 : 900);
  };

  const continueAfterRestoration = useCallback(() => {
    const r = restoration;
    const p = progressRef.current;
    const c = campaignRef.current;
    const m = modeRef.current;
    setRestoration(null);
    if (m === 'campaign' && r?.zoneDone) {
      if (send('ZONE_DONE')) {
        const zone = c.zones[Math.min(p.zoneIndex, c.zones.length - 1)];
        bus.emit('zone:complete', { zoneId: zone.id, final: !!zone.final });
      }
      return;
    }
    if (m === 'revisit' && sessionRef.current.revisitQueue.every(n => p.recovered[n])) { send('TO_TITLE'); return; }
    if (send('NEXT_TARGET')) {
      const zone = m === 'campaign' ? c.zones[Math.min(p.zoneIndex, c.zones.length - 1)] : undefined;
      beginTarget(p, zone, r?.target ?? null);
    }
  }, [restoration, send, beginTarget]);

  /** Called by the level-complete screen: bank the zone bonus and move on. */
  const finishZone = useCallback(() => {
    const p = progressRef.current;
    const c = campaignRef.current;
    const zone = c.zones[Math.min(p.zoneIndex, c.zones.length - 1)];
    let next = completeZone(p, zone.id, zoneBonus(p.hintsByZone[zone.id] ?? 0), c.zones.length);
    if (zone.final) next = { ...next, campaignComplete: true };
    const unlocked = evaluateAchievements(next, c);
    if (unlocked.length) next = { ...next, achievements: [...next.achievements, ...unlocked] };
    setProgress(next);
    progressRef.current = next;
    unlocked.forEach((id, i) => setTimeout(() => { bus.emit('achievement:unlock', { id }); setToast({ id, key: Date.now() + i }); }, 600 + i * 2600));
    if (zone.final) send('BEGIN_FINALE'); else send('NEXT_ZONE');
  }, [send]);

  const skipTarget = useCallback(() => {
    const s = sessionRef.current;
    if (!s.target) return;
    client.cancel();
    if (!canSend(machineRef.current.phase, 'SKIP_TARGET')) return;
    const p0 = progressRef.current;
    const m = modeRef.current;
    const p = m === 'campaign' || m === 'revisit' ? recordSkip(p0, s.target) : p0;
    setProgress(p);
    progressRef.current = p;
    bus.emit('memory:skipped', { target: s.target });
    send('SKIP_TARGET');
    const c = campaignRef.current;
    let zone: ZoneSpec | undefined = m === 'campaign' ? c.zones[Math.min(p.zoneIndex, c.zones.length - 1)] : undefined;
    // In Act III, top the sequence up if skips would leave too few memories to finish.
    if (zone?.final && p.finalQueue) {
      const remaining = p.finalQueue.filter(n => !p.finalRecovered.includes(n));
      if (remaining.length + p.finalRecovered.length < zone.required + 2) {
        const extra = c.classes.filter(n => !p.finalQueue!.includes(n)).slice(0, 2);
        const topped = { ...p, finalQueue: [...p.finalQueue, ...extra] };
        setProgress(topped);
        progressRef.current = topped;
      }
    }
    beginTarget(progressRef.current, zone, s.target);
    say('Leaving that one in the dark for now. It’s not going anywhere.', 'idle');
  }, [send, beginTarget, say, client]);

  const revealHint = useCallback(() => {
    const s = sessionRef.current;
    if (s.hintLevel >= 2 || !s.target) return;
    setSession(prev => ({ ...prev, hintLevel: prev.hintLevel + 1 }));
    if (modeRef.current === 'campaign' || modeRef.current === 'revisit') setProgress(p => recordHint(p, s.zoneId ?? 'practice'));
    bus.emit('hint:reveal', { level: s.hintLevel + 1 });
  }, []);

  const finishFinale = useCallback(() => {
    setProgress(p => ({ ...p, campaignComplete: true }));
    send('FINALE_DONE');
  }, [send]);

  const toTitle = useCallback(() => {
    client.cancel();
    setRestoration(null);
    setSession(emptySession());
    setMode('campaign');
    send('TO_TITLE');
  }, [send, client]);

  const resetSave = useCallback(() => {
    clearProgress();
    const fresh = newProgress();
    setProgress(fresh);
    progressRef.current = fresh;
    setHasSave(false);
  }, []);

  // Rush timer: when it runs out, the round ends after any pending answer.
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (mode !== 'rush' || !session.rushEndsAt) return;
    const id = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(id);
  }, [mode, session.rushEndsAt]);
  const rushLeft = mode === 'rush' && session.rushEndsAt ? Math.max(0, Math.ceil((session.rushEndsAt - now) / 1000)) : null;
  const rushOver = rushLeft === 0;
  useEffect(() => {
    if (!rushOver) return;
    client.cancel();
    setProgress(p => (session.rushScore > p.rushBest ? { ...p, rushBest: session.rushScore } : p));
  }, [rushOver]); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    phase: machine.phase as Phase, machine, send, model, modelError, loadModel, campaign, currentZone, progress, hasSave,
    settings, updateSettings, reducedMotion, mode, session, line, say, restoration, toast, integrity,
    startNewCampaign, continueCampaign, startPractice, startRevisit, markIntroSeen, enterZone, strokeStarted, clearDrawing,
    submitDrawing, continueAfterRestoration, finishZone, skipTarget, revealHint, finishFinale, toTitle, resetSave,
    rushLeft, rushOver,
  };
}

export type Game = ReturnType<typeof useGameState>;
const GameContext = createContext<Game | null>(null);

export function GameProvider({ children }: { children: ReactNode }) {
  const game = useGameState();
  return <GameContext.Provider value={game}>{children}</GameContext.Provider>;
}

export function useGame(): Game {
  const game = useContext(GameContext);
  if (!game) throw new Error('useGame must be used inside <GameProvider>.');
  return game;
}
