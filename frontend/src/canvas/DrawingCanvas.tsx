/**
 * The drawing surface. Starts blank except for faint registration marks;
 * nothing to trace. Strokes are kept as point lists (the source of truth) and
 * rendered as phosphor ink. Pointer handling avoids React re-renders while
 * drawing: only stroke start/end update the parent.
 */
import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from 'react';
import { CANVAS_SIZE, tracePath, type Point, type Stroke } from './strokes';
import { bus } from '../events/bus';

export interface DrawingCanvasHandle {
  strokes(): Stroke[];
  clear(): void;
  undo(): void;
  /** Stroke points in viewport coordinates (for the shard burst). */
  screenPoints(): [number, number][];
  load(strokes: Stroke[]): void;
}

interface Props {
  locked: boolean;
  tone: 'idle' | 'scanning' | 'recognized' | 'missed' | 'uncertain';
  onStrokeStart(): void;
  onChange(count: number): void;
  onPointer?(p: { x: number; y: number } | null): void;
  ariaLabel: string;
}

const INK: Record<Props['tone'], string> = {
  idle: '#f3ecd9', scanning: '#29e7ff', recognized: '#c8ff3d', missed: '#ff8a7a', uncertain: '#b8a6ff',
};

export const DrawingCanvas = forwardRef<DrawingCanvasHandle, Props>(function DrawingCanvas(props, ref) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const strokes = useRef<Stroke[]>([]);
  const active = useRef<{ id: number; stroke: Stroke } | null>(null);
  const live = useRef(props);
  live.current = props;
  const scanLine = useRef(0);

  const render = useCallback(() => {
    const el = canvas.current;
    if (!el) return;
    const ctx = el.getContext('2d')!;
    const scale = el.width / CANVAS_SIZE;
    ctx.clearRect(0, 0, el.width, el.height);
    // registration marks: corners and a centre cross, very faint
    ctx.strokeStyle = 'rgba(243,236,217,0.10)';
    ctx.lineWidth = 1.5 * scale;
    const m = 18 * scale, L = 22 * scale, W = el.width;
    ctx.beginPath();
    for (const [x, y, dx, dy] of [[m, m, 1, 1], [W - m, m, -1, 1], [m, W - m, 1, -1], [W - m, W - m, -1, -1]]) {
      ctx.moveTo(x, y + dy * L); ctx.lineTo(x, y); ctx.lineTo(x + dx * L, y);
    }
    ctx.moveTo(W / 2 - 6 * scale, W / 2); ctx.lineTo(W / 2 + 6 * scale, W / 2);
    ctx.moveTo(W / 2, W / 2 - 6 * scale); ctx.lineTo(W / 2, W / 2 + 6 * scale);
    ctx.stroke();
    const color = INK[live.current.tone];
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    // glow pass + core pass
    for (const [width, alpha] of [[16, 0.12], [7, 1]] as const) {
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = ctx.fillStyle = color;
      ctx.lineWidth = width * scale;
      for (const s of strokes.current) tracePath(ctx, s, scale);
      if (active.current) tracePath(ctx, active.current.stroke, scale);
    }
    ctx.globalAlpha = 1;
    if (live.current.tone === 'scanning') {
      const y = (scanLine.current % 1) * el.height;
      const grad = ctx.createLinearGradient(0, y - 40 * scale, 0, y);
      grad.addColorStop(0, 'rgba(41,231,255,0)');
      grad.addColorStop(1, 'rgba(41,231,255,0.35)');
      ctx.fillStyle = grad;
      ctx.fillRect(0, y - 40 * scale, el.width, 40 * scale);
      ctx.fillStyle = 'rgba(41,231,255,0.9)';
      ctx.fillRect(0, y, el.width, 2 * scale);
    }
  }, []);

  // keep the backing store sharp at any CSS size
  useEffect(() => {
    const el = canvas.current!;
    const fit = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const size = Math.max(200, Math.round(el.clientWidth * dpr));
      if (el.width !== size) { el.width = el.height = size; render(); }
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    return () => ro.disconnect();
  }, [render]);

  // scan animation only while scanning
  useEffect(() => {
    render();
    if (props.tone !== 'scanning') return;
    let frame = 0;
    const start = performance.now();
    const loop = (now: number) => { scanLine.current = (now - start) / 900; render(); frame = requestAnimationFrame(loop); };
    frame = requestAnimationFrame(loop);
    return () => { cancelAnimationFrame(frame); render(); };
  }, [props.tone, render]);

  const toLocal = (e: PointerEvent | React.PointerEvent): Point => {
    const rect = canvas.current!.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * CANVAS_SIZE;
    const y = ((e.clientY - rect.top) / rect.height) * CANVAS_SIZE;
    return [Math.max(0, Math.min(CANVAS_SIZE, x)), Math.max(0, Math.min(CANVAS_SIZE, y))];
  };

  const onDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (live.current.locked || active.current || (e.pointerType === 'mouse' && e.button !== 0)) return;
    e.preventDefault();
    canvas.current!.setPointerCapture(e.pointerId);
    const p = toLocal(e);
    active.current = { id: e.pointerId, stroke: { points: [p] } };
    live.current.onStrokeStart();
    bus.emit('draw:stroke-start', { x: p[0] / CANVAS_SIZE, y: p[1] / CANVAS_SIZE });
    render();
  };

  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const p = toLocal(e);
    live.current.onPointer?.({ x: p[0] / CANVAS_SIZE * 2 - 1, y: p[1] / CANVAS_SIZE * 2 - 1 });
    const a = active.current;
    if (!a || a.id !== e.pointerId) return;
    const events = (e.nativeEvent as PointerEvent).getCoalescedEvents?.() ?? [e.nativeEvent];
    for (const ev of events) {
      const q = toLocal(ev);
      const last = a.stroke.points[a.stroke.points.length - 1];
      if (Math.hypot(q[0] - last[0], q[1] - last[1]) >= 1.2) a.stroke.points.push(q);
    }
    render();
  };

  const finish = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const a = active.current;
    if (!a || a.id !== e.pointerId) return;
    active.current = null;
    strokes.current = [...strokes.current, a.stroke];
    live.current.onChange(strokes.current.length);
    bus.emit('draw:stroke-end', { strokes: strokes.current.length });
    render();
  };

  useImperativeHandle(ref, () => ({
    strokes: () => strokes.current.map(s => ({ points: s.points.slice() })),
    clear: () => { strokes.current = []; active.current = null; live.current.onChange(0); render(); },
    undo: () => {
      if (!strokes.current.length) return;
      strokes.current = strokes.current.slice(0, -1);
      live.current.onChange(strokes.current.length);
      bus.emit('draw:undo', {});
      render();
    },
    screenPoints: () => {
      const rect = canvas.current!.getBoundingClientRect();
      const k = rect.width / CANVAS_SIZE;
      return strokes.current.flatMap(s => s.points.map(([x, y]) => [rect.left + x * k, rect.top + y * k] as [number, number]));
    },
    load: (next: Stroke[]) => { strokes.current = next; live.current.onChange(next.length); render(); },
  }), [render]);

  return (
    <canvas
      ref={canvas}
      className={`draw-surface tone-${props.tone}${props.locked ? ' is-locked' : ''}`}
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={finish}
      onPointerCancel={finish}
      onPointerLeave={() => live.current.onPointer?.(null)}
      onContextMenu={e => e.preventDefault()}
      role="img"
      aria-label={props.ariaLabel}
    />
  );
});
