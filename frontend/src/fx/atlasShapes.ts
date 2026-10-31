/**
 * The pre-corruption Atlas, drawn in code: simple, deliberately wobbly line
 * icons in a unit box, standing in for "everyone's drawings" in the opening.
 * They're generated from geometry plus seeded jitter, not from any dataset.
 */
import { rng } from '../game/campaign';

export type Poly = [number, number][];

const circle = (cx: number, cy: number, r: number, n = 22, a0 = 0, a1 = Math.PI * 2): Poly =>
  Array.from({ length: n + 1 }, (_, i) => { const a = a0 + (a1 - a0) * (i / n); return [cx + Math.cos(a) * r, cy + Math.sin(a) * r]; });

const SHAPES: Record<string, () => Poly[]> = {
  house: () => [[[0.15, 0.95], [0.15, 0.45], [0.85, 0.45], [0.85, 0.95], [0.15, 0.95]], [[0.08, 0.5], [0.5, 0.1], [0.92, 0.5]], [[0.42, 0.95], [0.42, 0.7], [0.58, 0.7], [0.58, 0.95]]],
  sun: () => [circle(0.5, 0.5, 0.22), ...Array.from({ length: 8 }, (_, i) => { const a = (i / 8) * Math.PI * 2; return [[0.5 + Math.cos(a) * 0.32, 0.5 + Math.sin(a) * 0.32], [0.5 + Math.cos(a) * 0.46, 0.5 + Math.sin(a) * 0.46]] as Poly; })],
  tree: () => [[[0.45, 0.98], [0.45, 0.6]], [[0.55, 0.98], [0.55, 0.6]], circle(0.5, 0.38, 0.3, 26)],
  star: () => [Array.from({ length: 11 }, (_, i) => { const a = -Math.PI / 2 + (i * 4 * Math.PI) / 5; return [0.5 + Math.cos(a) * 0.45, 0.52 + Math.sin(a) * 0.45] as [number, number]; })],
  boat: () => [[[0.1, 0.7], [0.9, 0.7], [0.75, 0.88], [0.25, 0.88], [0.1, 0.7]], [[0.5, 0.7], [0.5, 0.08], [0.85, 0.62], [0.5, 0.62]]],
  fish: () => [circle(0.45, 0.5, 0.28, 24, 0.5, Math.PI * 2 - 0.5), [[0.7, 0.36], [0.95, 0.22], [0.95, 0.78], [0.7, 0.64]], circle(0.3, 0.45, 0.03, 6)],
  cloud: () => [[...circle(0.3, 0.6, 0.16, 10, Math.PI * 0.5, Math.PI * 1.5), ...circle(0.5, 0.45, 0.2, 12, Math.PI, Math.PI * 2), ...circle(0.72, 0.6, 0.16, 10, Math.PI * 1.5, Math.PI * 2.5), [0.3, 0.76]]],
  mountain: () => [[[0.02, 0.9], [0.3, 0.35], [0.45, 0.6], [0.62, 0.15], [0.98, 0.9], [0.02, 0.9]]],
  moon: () => [[...circle(0.5, 0.5, 0.38, 20, Math.PI * 0.35, Math.PI * 1.65), ...circle(0.68, 0.5, 0.3, 16, Math.PI * 1.55, Math.PI * 0.45).reverse()]],
  flower: () => [circle(0.5, 0.35, 0.08, 10), ...Array.from({ length: 6 }, (_, i) => circle(0.5 + Math.cos(i * 1.047) * 0.17, 0.35 + Math.sin(i * 1.047) * 0.17, 0.09, 10)), [[0.5, 0.45], [0.5, 0.98]], [[0.5, 0.75], [0.66, 0.66]]],
  umbrella: () => [[...circle(0.5, 0.48, 0.42, 20, Math.PI, Math.PI * 2), [0.08, 0.48]], [[0.5, 0.48], [0.5, 0.88], [0.42, 0.92], [0.38, 0.86]]],
  bicycle: () => [circle(0.22, 0.68, 0.18), circle(0.78, 0.68, 0.18), [[0.22, 0.68], [0.42, 0.4], [0.68, 0.4], [0.78, 0.68]], [[0.42, 0.4], [0.5, 0.68], [0.68, 0.4]], [[0.68, 0.4], [0.64, 0.28], [0.74, 0.26]]],
};

export const ATLAS_KINDS = Object.keys(SHAPES);

export function atlasShape(kind: string, seed: number): Poly[] {
  const random = rng(seed);
  const base = (SHAPES[kind] ?? SHAPES.star)();
  // Hand-drawn wobble: low-frequency jitter along each line.
  return base.map(poly => {
    const out: Poly = [];
    for (let i = 0; i < poly.length; i++) {
      const [x, y] = poly[i];
      out.push([x + (random() - 0.5) * 0.035, y + (random() - 0.5) * 0.035]);
      if (i < poly.length - 1) {
        const [nx, ny] = poly[i + 1];
        out.push([(x + nx) / 2 + (random() - 0.5) * 0.03, (y + ny) / 2 + (random() - 0.5) * 0.03]);
      }
    }
    return out;
  });
}
