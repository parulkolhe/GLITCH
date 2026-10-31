/**
 * Imperative hooks into the world canvas for one-shot effects. Screens call
 * these with screen coordinates; the world layer consumes them on its next frame.
 */
export interface ShardRequest { points: [number, number][]; target: [number, number]; color: string }
export interface PulseRequest { x: number; y: number; color: string; strength: number }

type Listener = { shards(r: ShardRequest): void; pulse(r: PulseRequest): void; glitch(amount: number): void };
let listener: Listener | null = null;

export const fx = {
  attach(l: Listener) { listener = l; return () => { if (listener === l) listener = null; }; },
  /** Burst the drawing into particles that stream to `target` (usually GLITCH's core). */
  shards(points: [number, number][], target: [number, number], color = '#c8ff3d') { listener?.shards({ points, target, color }); },
  pulse(x: number, y: number, color = '#c8ff3d', strength = 1) { listener?.pulse({ x, y, color, strength }); },
  /** Momentary signal tear across the screen (0..1). */
  glitch(amount = 0.6) { listener?.glitch(amount); },
};
