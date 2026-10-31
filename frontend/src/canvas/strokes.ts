/**
 * Strokes are the source of truth for a drawing. The on-screen canvas renders
 * them as glowing ink on a dark surface; the PNG sent to the CNN is rendered
 * separately from the same strokes as dark ink on white, at a fixed 600x600,
 * because that is what backend/ml/preprocessing.py was written and tested for
 * (any non-white pixel is ink; crop, centre, dilate, resize to 28x28, invert).
 */

export const CANVAS_SIZE = 600;     // logical drawing units (square)
export const EXPORT_SIZE = 600;     // PNG side sent to the API
export const EXPORT_LINE_WIDTH = 10; // at 600 px; the backend dilates further before downsampling

export type Point = [number, number];
export interface Stroke { points: Point[] }

export function strokeLength(stroke: Stroke): number {
  let total = 0;
  for (let i = 1; i < stroke.points.length; i++) {
    const [x0, y0] = stroke.points[i - 1];
    const [x1, y1] = stroke.points[i];
    total += Math.hypot(x1 - x0, y1 - y0);
  }
  return total;
}

export function inkAmount(strokes: Stroke[]): number {
  return strokes.reduce((sum, s) => sum + Math.max(strokeLength(s), s.points.length ? 1 : 0), 0);
}

export function bounds(strokes: Stroke[]): { x: number; y: number; w: number; h: number } | null {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const s of strokes) for (const [x, y] of s.points) {
    if (x < minX) minX = x; if (y < minY) minY = y; if (x > maxX) maxX = x; if (y > maxY) maxY = y;
  }
  return minX === Infinity ? null : { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

/** Ramer-Douglas-Peucker simplification, for compact saves of recovered drawings. */
export function simplify(points: Point[], epsilon = 1.6): Point[] {
  if (points.length < 3) return points.slice();
  const keep = new Uint8Array(points.length);
  keep[0] = keep[points.length - 1] = 1;
  const stack: [number, number][] = [[0, points.length - 1]];
  while (stack.length) {
    const [start, end] = stack.pop()!;
    const [ax, ay] = points[start];
    const [bx, by] = points[end];
    const dx = bx - ax, dy = by - ay;
    const len = Math.hypot(dx, dy);
    let best = -1, bestDist = 0;
    for (let i = start + 1; i < end; i++) {
      const [px, py] = points[i];
      // closed loops (start == end) have no chord: measure distance from the start point instead
      const dist = len < 1e-6 ? Math.hypot(px - ax, py - ay) : Math.abs(dy * px - dx * py + bx * ay - by * ax) / len;
      if (dist > bestDist) { bestDist = dist; best = i; }
    }
    if (best > 0 && bestDist > epsilon) {
      keep[best] = 1;
      stack.push([start, best], [best, end]);
    }
  }
  return points.filter((_, i) => keep[i]);
}

export function compactStrokes(strokes: Stroke[]): number[][] {
  return strokes.map(s => simplify(s.points).flatMap(([x, y]) => [Math.round(x), Math.round(y)]));
}

export function expandStrokes(compact: number[][]): Stroke[] {
  return compact.map(flat => {
    const points: Point[] = [];
    for (let i = 0; i + 1 < flat.length; i += 2) points.push([flat[i], flat[i + 1]]);
    return { points };
  });
}

/** Minimal 2D context surface used by the renderers (lets tests pass a fake). */
export interface Ctx2D {
  beginPath(): void; moveTo(x: number, y: number): void; lineTo(x: number, y: number): void;
  quadraticCurveTo(cx: number, cy: number, x: number, y: number): void;
  arc(x: number, y: number, r: number, a0: number, a1: number): void;
  stroke(): void; fill(): void;
  lineWidth: number; lineCap: CanvasLineCap; lineJoin: CanvasLineJoin;
  strokeStyle: string | CanvasGradient | CanvasPattern; fillStyle: string | CanvasGradient | CanvasPattern;
}

/** Smooth path through the points (midpoint quadratic curves), dots for taps. */
export function tracePath(ctx: Ctx2D, stroke: Stroke, scale = 1, ox = 0, oy = 0) {
  const pts = stroke.points;
  if (!pts.length) return;
  const X = (p: Point) => ox + p[0] * scale;
  const Y = (p: Point) => oy + p[1] * scale;
  ctx.beginPath();
  if (pts.length === 1) {
    ctx.arc(X(pts[0]), Y(pts[0]), ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  ctx.moveTo(X(pts[0]), Y(pts[0]));
  for (let i = 1; i < pts.length - 1; i++) {
    const mx = (X(pts[i]) + X(pts[i + 1])) / 2;
    const my = (Y(pts[i]) + Y(pts[i + 1])) / 2;
    ctx.quadraticCurveTo(X(pts[i]), Y(pts[i]), mx, my);
  }
  ctx.lineTo(X(pts[pts.length - 1]), Y(pts[pts.length - 1]));
  ctx.stroke();
}

/**
 * Render the strokes the way the CNN pipeline expects: black ink on an opaque
 * white square. Returns a PNG data URL.
 */
export function exportPng(strokes: Stroke[], doc: Document = document): string {
  const canvas = doc.createElement('canvas');
  canvas.width = canvas.height = EXPORT_SIZE;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D is unavailable.');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, EXPORT_SIZE, EXPORT_SIZE);
  ctx.strokeStyle = ctx.fillStyle = '#000000';
  ctx.lineWidth = EXPORT_LINE_WIDTH * (EXPORT_SIZE / CANVAS_SIZE);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const stroke of strokes) tracePath(ctx, stroke, EXPORT_SIZE / CANVAS_SIZE);
  return canvas.toDataURL('image/png');
}
