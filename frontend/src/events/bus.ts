/**
 * Central game event bus. Gameplay code emits facts ("a prediction came back",
 * "a memory was restored"); the audio engine, the robot and the world FX
 * subscribe. Nothing listens to raw pointer events except the canvas.
 */
import type { PredictionResult } from '../services/predictionClient';

export type GameEvents = {
  'ui:click': { variant?: 'soft' | 'confirm' | 'back' };
  'ui:hover': {};
  'boot:start': {};
  'boot:relay': { index: number };
  'boot:signature': {};
  'cinematic:beat': { mood?: string; fx?: string };
  'core:charge': { level: number };
  'core:online': {};
  'zone:enter': { zoneId: string; act: number };
  'draw:stroke-start': { x: number; y: number };
  'draw:stroke-end': { strokes: number };
  'draw:clear': {};
  'draw:undo': {};
  'predict:start': { target: string };
  'predict:result': { result: PredictionResult; target: string; attempt: number };
  'predict:error': { message: string };
  'memory:recognized': { target: string; streak: number; points: number; firstTry: boolean };
  'memory:missed': { target: string; predicted: string; uncertain: boolean; attempt: number };
  'memory:shard': { index: number };
  'memory:skipped': { target: string };
  'hint:reveal': { level: number };
  'streak:milestone': { streak: number };
  'achievement:unlock': { id: string };
  'zone:complete': { zoneId: string; final: boolean };
  'finale:phase': { phase: 'noise' | 'converge' | 'reboot' | 'world' | 'credits' };
  'settings:changed': {};
};

type Handler<T> = (payload: T) => void;

export class EventBus<E extends Record<string, unknown>> {
  private handlers = new Map<keyof E, Set<Handler<any>>>();

  on<K extends keyof E>(type: K, handler: Handler<E[K]>): () => void {
    let set = this.handlers.get(type);
    if (!set) this.handlers.set(type, (set = new Set()));
    set.add(handler);
    return () => set!.delete(handler);
  }

  emit<K extends keyof E>(type: K, payload: E[K]): void {
    const set = this.handlers.get(type);
    if (!set) return;
    for (const handler of [...set]) {
      try { handler(payload); } catch (error) { console.error(`[bus] ${String(type)} handler failed`, error); }
    }
  }

  clear(): void { this.handlers.clear(); }
}

export const bus = new EventBus<GameEvents>();
