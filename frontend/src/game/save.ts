/**
 * Local persistence. Browser storage can be missing or throw (private mode,
 * blocked storage), so every access is guarded and the game always runs
 * without it; it just won't remember.
 */
import { newProgress, SAVE_VERSION, type Progress } from './progress';

export const SAVE_KEY = 'glitch.campaign.v2';
export const SETTINGS_KEY = 'glitch.settings.v1';

export interface Settings {
  masterVolume: number;   // 0..1
  musicVolume: number;
  sfxVolume: number;
  muted: boolean;
  reducedMotion: 'system' | 'on' | 'off';
  drawingSounds: boolean;
  showModelDetails: boolean;  // show top-k bars and the model's 28x28 input after each attempt
  textSpeed: 'normal' | 'fast' | 'instant';
}

export const DEFAULT_SETTINGS: Settings = {
  masterVolume: 0.8, musicVolume: 0.55, sfxVolume: 0.8, muted: false, reducedMotion: 'system',
  drawingSounds: true, showModelDetails: true, textSpeed: 'normal',
};

export interface StorageLike { getItem(k: string): string | null; setItem(k: string, v: string): void; removeItem(k: string): void }

function storage(): StorageLike | null {
  try { return typeof localStorage !== 'undefined' ? localStorage : null; } catch { return null; }
}

const num = (v: unknown, lo: number, hi: number, d: number) =>
  typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d;

/** Validate a parsed save; returns null for anything unusable so a corrupt save can't crash the game. */
export function validateProgress(raw: unknown): Progress | null {
  if (!raw || typeof raw !== 'object') return null;
  const r = raw as Record<string, any>;
  if (r.version !== SAVE_VERSION) return null;
  const base = newProgress(typeof r.seed === 'number' ? r.seed : undefined);
  const arr = (v: unknown) => (Array.isArray(v) ? v.filter(x => typeof x === 'string') : []);
  const recovered: Progress['recovered'] = {};
  if (r.recovered && typeof r.recovered === 'object') {
    for (const [name, m] of Object.entries(r.recovered as Record<string, any>)) {
      if (!m || typeof m !== 'object' || !Array.isArray(m.strokes)) continue;
      recovered[name] = {
        zoneId: String(m.zoneId ?? ''), attempts: num(m.attempts, 1, 999, 1), hintsUsed: num(m.hintsUsed, 0, 99, 0),
        clarity: num(m.clarity, 0, 1, 0), points: num(m.points, 0, 1e5, 0),
        strokes: m.strokes.filter((s: unknown) => Array.isArray(s)).map((s: unknown[]) => s.filter(n => typeof n === 'number') as number[]),
        at: num(m.at, 0, 1e15, 0), modelVersion: String(m.modelVersion ?? ''),
      };
    }
  }
  return {
    ...base,
    startedAt: num(r.startedAt, 0, 1e15, base.startedAt),
    zoneIndex: num(r.zoneIndex, 0, 99, 0),
    zonesCompleted: arr(r.zonesCompleted), recovered, skipped: arr(r.skipped), seen: arr(r.seen),
    finalQueue: Array.isArray(r.finalQueue) ? arr(r.finalQueue) : null, finalRecovered: arr(r.finalRecovered),
    score: num(r.score, 0, 1e7, 0), xp: num(r.xp, 0, 1e7, 0), fragments: num(r.fragments, 0, 1e5, 0),
    streak: num(r.streak, 0, 1e4, 0), bestStreak: num(r.bestStreak, 0, 1e4, 0), attempts: num(r.attempts, 0, 1e6, 0),
    hintsUsed: num(r.hintsUsed, 0, 1e5, 0),
    hintsByZone: r.hintsByZone && typeof r.hintsByZone === 'object' ? Object.fromEntries(Object.entries(r.hintsByZone).filter(([, v]) => typeof v === 'number')) as Record<string, number> : {},
    confusions: Array.isArray(r.confusions) ? r.confusions.filter((c: any) => c && typeof c.target === 'string' && typeof c.predicted === 'string' && typeof c.count === 'number') : [],
    uncertainCount: num(r.uncertainCount, 0, 1e6, 0), achievements: arr(r.achievements),
    campaignComplete: r.campaignComplete === true, introSeen: r.introSeen === true, rushBest: num(r.rushBest, 0, 1e7, 0),
  };
}

export function loadProgress(store: StorageLike | null = storage()): Progress | null {
  if (!store) return null;
  try {
    const text = store.getItem(SAVE_KEY);
    return text ? validateProgress(JSON.parse(text)) : null;
  } catch { return null; }
}

export function saveProgress(p: Progress, store: StorageLike | null = storage()): boolean {
  if (!store) return false;
  try { store.setItem(SAVE_KEY, JSON.stringify(p)); return true; } catch { return false; }
}

export function clearProgress(store: StorageLike | null = storage()): void {
  try { store?.removeItem(SAVE_KEY); } catch { /* ignore */ }
}

export function loadSettings(store: StorageLike | null = storage()): Settings {
  try {
    const raw = store?.getItem(SETTINGS_KEY);
    if (!raw) return { ...DEFAULT_SETTINGS };
    const r = JSON.parse(raw) ?? {};
    return {
      masterVolume: num(r.masterVolume, 0, 1, DEFAULT_SETTINGS.masterVolume),
      musicVolume: num(r.musicVolume, 0, 1, DEFAULT_SETTINGS.musicVolume),
      sfxVolume: num(r.sfxVolume, 0, 1, DEFAULT_SETTINGS.sfxVolume),
      muted: r.muted === true,
      reducedMotion: ['system', 'on', 'off'].includes(r.reducedMotion) ? r.reducedMotion : 'system',
      drawingSounds: r.drawingSounds !== false,
      showModelDetails: r.showModelDetails !== false,
      textSpeed: ['normal', 'fast', 'instant'].includes(r.textSpeed) ? r.textSpeed : 'normal',
    };
  } catch { return { ...DEFAULT_SETTINGS }; }
}

export function saveSettings(s: Settings, store: StorageLike | null = storage()): void {
  try { store?.setItem(SETTINGS_KEY, JSON.stringify(s)); } catch { /* ignore */ }
}
