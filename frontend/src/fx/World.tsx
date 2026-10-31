/**
 * The world layer: one full-screen Canvas 2D renderer, one rAF loop.
 *
 * Layers, back to front: sky, stars, the scene (Atlas icons, the signal flood,
 * constellations made of the player's recovered drawings), the horizon grid,
 * shard particles and pulses, then corruption tears whose strength is the
 * campaign's real corruption level (1 - integrity).
 *
 * With reduced motion the loop throttles to a near-static redraw, with no
 * scrolling, shaking or tearing.
 */
import { useEffect, useRef } from 'react';
import { atlasShape, ATLAS_KINDS, type Poly } from './atlasShapes';
import { fx, type ShardRequest, type PulseRequest } from './fxController';
import { expandStrokes } from '../canvas/strokes';
import { rng } from '../game/campaign';

export type WorldScene = 'title' | 'cinematic' | 'zone' | 'finale';
export type GridStyle = 'flat' | 'waves' | 'rings' | 'shards' | 'spiral' | 'core';

export interface Constellation { name: string; strokes: number[][]; addedAt?: number }

export interface WorldProps {
  scene: WorldScene;
  beat?: string;          // cinematic fx key
  corruption: number;     // 0..1
  hue: number;            // zone hue in degrees
  grid: GridStyle;
  constellations: Constellation[];
  reducedMotion: boolean;
}

interface Particle { x: number; y: number; vx: number; vy: number; tx: number; ty: number; life: number; color: string; size: number; delay: number }
interface Pulse extends PulseRequest { born: number }

const hash = (s: string) => { let h = 2166136261; for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619); return h >>> 0; };

export function World(props: WorldProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const live = useRef(props);
  const beatStart = useRef(performance.now());
  const lastBeat = useRef<string | undefined>(undefined);
  if (props.beat !== lastBeat.current) { lastBeat.current = props.beat; beatStart.current = performance.now(); }
  live.current = props;

  useEffect(() => {
    const el = canvas.current!;
    const ctx = el.getContext('2d', { alpha: false })!;
    let w = 0, h = 0, dpr = 1, frame = 0, lastDraw = 0;
    const particles: Particle[] = [];
    const pulses: Pulse[] = [];
    let tear = 0;

    const random = rng(7);
    const stars = Array.from({ length: 220 }, () => ({ x: random(), y: random() * 0.72, r: random() * 1.3 + 0.2, p: random() * 6.28, s: 0.4 + random() * 1.6 }));
    // the Atlas: icons on a jittered 8x3 grid so they read as a world, not a pile
    const atlas = Array.from({ length: 24 }, (_, i) => {
      const col = i % 8, row = Math.floor(i / 8);
      return {
        kind: ATLAS_KINDS[(i * 5) % ATLAS_KINDS.length], polys: atlasShape(ATLAS_KINDS[(i * 5) % ATLAS_KINDS.length], 100 + i),
        x: (col + 0.5 + (row % 2) * 0.35 + (random() - 0.5) * 0.3) / 8.35, y: 0.12 + row * 0.27 + (random() - 0.5) * 0.08,
        s: 0.05 + random() * 0.03, delay: (col + row * 3) * 0.12, vy: 0, rot: 0, vr: (random() - 0.5) * 0.6,
      };
    });
    const flood = Array.from({ length: 900 }, () => ({ x: random(), y: -random() * 2.2, v: 0.3 + random() * 1.4, len: 4 + random() * 18 }));
    const strokeCache = new Map<string, Poly[]>();

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      w = el.clientWidth; h = el.clientHeight;
      el.width = Math.max(1, Math.floor(w * dpr)); el.height = Math.max(1, Math.floor(h * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(el);

    const detach = fx.attach({
      shards: (r: ShardRequest) => {
        const step = Math.max(1, Math.floor(r.points.length / 160));
        for (let i = 0; i < r.points.length; i += step) {
          const [x, y] = r.points[i];
          const a = Math.random() * Math.PI * 2;
          const sp = 1 + Math.random() * 4;
          particles.push({ x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - 1, tx: r.target[0], ty: r.target[1], life: 1,
            color: r.color, size: 1 + Math.random() * 2.2, delay: Math.random() * 18 });
        }
      },
      pulse: (r: PulseRequest) => pulses.push({ ...r, born: performance.now() }),
      glitch: (amount: number) => { tear = Math.max(tear, amount); },
    });

    const polysFor = (c: Constellation): Poly[] => {
      let polys = strokeCache.get(c.name);
      if (!polys) {
        polys = expandStrokes(c.strokes).map(s => s.points.map(([x, y]) => [x / 600, y / 600] as [number, number]));
        strokeCache.set(c.name, polys);
      }
      return polys;
    };

    const drawPolys = (polys: Poly[], cx: number, cy: number, size: number, color: string, progress = 1, alpha = 1, rot = 0, jitter = 0) => {
      const total = polys.reduce((n, p) => n + p.length, 0);
      let budget = Math.floor(total * progress);
      ctx.save();
      ctx.translate(cx, cy);
      if (rot) ctx.rotate(rot);
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      for (const pass of [0, 1]) {
        let b = budget;
        ctx.strokeStyle = color;
        ctx.globalAlpha = alpha * (pass === 0 ? 0.18 : 0.95);
        ctx.lineWidth = pass === 0 ? 6 : 1.6;
        for (const poly of polys) {
          if (b <= 1) break;
          ctx.beginPath();
          const n = Math.min(poly.length, b);
          for (let i = 0; i < n; i++) {
            const jx = jitter ? (Math.random() - 0.5) * jitter : 0;
            const px = (poly[i][0] - 0.5) * size + jx, py = (poly[i][1] - 0.5) * size;
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }
          ctx.stroke();
          b -= poly.length;
        }
      }
      budget = 0;
      ctx.restore();
      ctx.globalAlpha = 1;
    };

    const horizonY = () => h * 0.66;

    const drawGrid = (t: number, style: GridStyle, hue: number, corruption: number, brightness: number, still: boolean) => {
      const hy = horizonY();
      const vx = w / 2;
      const scroll = still ? 0 : (t * 0.06) % 1;
      const base = `hsla(${hue}, 95%, 62%,`;
      ctx.lineWidth = 1;
      // horizon line
      ctx.strokeStyle = `${base} ${0.45 * brightness})`;
      ctx.beginPath(); ctx.moveTo(0, hy); ctx.lineTo(w, hy); ctx.stroke();
      const rows = 16;
      for (let i = 0; i < rows; i++) {
        const z = (i + scroll) / rows;
        const y = hy + Math.pow(z, 2.2) * (h - hy);
        const a = 0.08 + z * 0.35;
        ctx.strokeStyle = `${base} ${a * brightness})`;
        ctx.beginPath();
        if (style === 'waves') {
          for (let x = 0; x <= w; x += 16) {
            const yy = y + Math.sin(x * 0.012 + t * 1.2 + i) * 8 * z;
            if (x === 0) ctx.moveTo(x, yy); else ctx.lineTo(x, yy);
          }
        } else if (style === 'rings' || style === 'core' || style === 'spiral') {
          const rx = (0.1 + z) * w * 0.7;
          const ry = (y - hy) * 0.9 + 2;
          const twist = style === 'spiral' ? t * 0.3 + i * 0.25 : 0;
          ctx.ellipse(vx, hy, rx, ry, 0, Math.PI + twist * 0.02, Math.PI * 2 - twist * 0.02, true);
        } else if (style === 'shards') {
          for (let x = 0; x < w; x += 70) {
            if ((hash(`${i}-${x}`) % 5) === 0) continue;
            const off = ((hash(`${x}`) % 7) - 3) * z;
            ctx.moveTo(x + 6, y + off); ctx.lineTo(x + 64, y + off);
          }
        } else {
          ctx.moveTo(0, y); ctx.lineTo(w, y);
        }
        ctx.stroke();
      }
      const cols = style === 'core' ? 36 : 22;
      for (let j = -cols; j <= cols; j++) {
        const x0 = vx + j * (w / cols) * 0.06;
        const x1 = vx + j * (w / cols) * 1.6;
        const wob = corruption > 0.3 && !still ? Math.sin(t * 3 + j) * corruption * 10 : 0;
        ctx.strokeStyle = `${base} ${(0.05 + 0.18 * (1 - Math.abs(j) / cols)) * brightness})`;
        ctx.beginPath(); ctx.moveTo(x0, hy); ctx.lineTo(x1 + wob, h); ctx.stroke();
      }
    };

    const loop = (now: number) => {
      frame = requestAnimationFrame(loop);
      const p = live.current;
      const still = p.reducedMotion;
      if (still && now - lastDraw < 250 && !particles.length && !pulses.length) return;
      lastDraw = now;
      const t = now / 1000;
      const bt = (now - beatStart.current) / 1000;
      const beat = p.scene === 'cinematic' || p.scene === 'finale' ? p.beat : undefined;
      let corruption = p.corruption;
      if (beat === 'atlas' || beat === 'atlas-draw') corruption = 0;
      if (beat === 'overflow') corruption = Math.min(1, bt / 3);
      if (beat === 'shatter' || beat === 'silence' || beat === 'boot' || beat === 'flicker') corruption = 1;

      // sky
      const sky = ctx.createLinearGradient(0, 0, 0, h);
      sky.addColorStop(0, '#04050a');
      sky.addColorStop(0.62, `hsl(${p.hue + 160}, 45%, ${beat === 'silence' ? 2 : 7}%)`);
      sky.addColorStop(1, '#04050a');
      ctx.fillStyle = sky;
      ctx.fillRect(0, 0, w, h);

      // stars
      if (beat !== 'silence') {
        for (const s of stars) {
          const tw = still ? 0.7 : 0.5 + 0.5 * Math.sin(t * s.s + s.p);
          ctx.fillStyle = `rgba(243,236,217,${0.15 + 0.55 * tw * (1 - corruption * 0.5)})`;
          ctx.fillRect(s.x * w, s.y * h, s.r, s.r);
        }
      }

      // scene content
      const skyH = horizonY();
      if (beat === 'atlas' || beat === 'atlas-draw' || beat === 'signal' || beat === 'overflow' || beat === 'shatter') {
        for (const a of atlas) {
          const size = a.s * Math.min(w, h) * 2.2;
          const prog = beat === 'atlas' ? Math.min(1, Math.max(0, (bt - a.delay * 0.6) / 1.4)) : 1;
          let x = a.x * w, y = a.y * skyH;
          let rot = 0;
          let alpha = 0.9;
          let jitter = 0;
          if (beat === 'overflow') { jitter = bt * 3; x += (Math.random() - 0.5) * bt * 4; }
          if (beat === 'shatter') {
            const fall = Math.max(0, bt - a.delay * 0.2);
            y += fall * fall * 140; rot = a.vr * fall; alpha = Math.max(0, 1 - fall * 0.5);
          }
          if (alpha <= 0) continue;
          const color = (hash(a.kind) % 3) === 0 ? '#c8ff3d' : (hash(a.kind) % 3) === 1 ? '#29e7ff' : '#f3ecd9';
          drawPolys(a.polys, x, y, size, color, prog, alpha, rot, jitter);
        }
      }
      if (beat === 'signal' || beat === 'overflow') {
        const n = Math.min(flood.length, Math.floor(bt * 260));
        ctx.strokeStyle = 'rgba(41,231,255,0.55)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          const f = flood[i];
          const y = ((f.y + bt * f.v) % 1.3) * h;
          const x = f.x * w + Math.sin(i) * 20;
          ctx.moveTo(x, y); ctx.lineTo(x, y + f.len);
        }
        ctx.stroke();
      }
      if (beat === 'shatter' && bt < 0.25) {
        ctx.fillStyle = `rgba(243,236,217,${1 - bt * 4})`;
        ctx.fillRect(0, 0, w, h);
      }

      // the player's recovered drawings as constellations
      if (p.scene !== 'cinematic' || beat === 'world' || beat === 'converge' || beat === 'reboot') {
        const finale = p.scene === 'finale';
        const n = p.constellations.length;
        p.constellations.forEach((c, i) => {
          const hsh = hash(c.name);
          let x: number, y: number, size: number;
          if (finale && beat === 'world') {
            // the restored Atlas: a wide, ordered arc of every recovered drawing
            const cols = Math.max(6, Math.ceil(Math.sqrt(n * 2.2)));
            const row = Math.floor(i / cols), col = i % cols;
            x = ((col + 0.5 + (row % 2) * 0.5) / (cols + 0.5)) * w;
            y = skyH * (0.14 + row * 0.2) + Math.sin(col * 0.9) * 10;
            size = Math.min(w / cols, skyH * 0.2) * 0.85;
          } else {
            x = (0.05 + (hsh % 1000) / 1000 * 0.9) * w;
            y = (0.06 + ((hsh >> 10) % 1000) / 1000 * 0.48) * skyH;
            size = Math.min(w, h) * 0.075;
          }
          if (finale && beat === 'converge') {
            const k = Math.min(1, bt / 3.2);
            x += (w / 2 - x) * k * k; y += (skyH * 0.5 - y) * k * k; size *= 1 - k * 0.7;
          }
          if (finale && beat === 'reboot') return;
          const age = c.addedAt ? (now - c.addedAt) / 1000 : 99;
          const prog = Math.min(1, age / 1.6);
          const drift = still ? 0 : Math.sin(t * 0.4 + hsh) * 3;
          const alpha = finale ? 1 : p.scene === 'zone' ? (age < 3 ? 0.9 : 0.32) : 0.7;
          drawPolys(polysFor(c), x, y + drift, size * (p.scene === 'zone' ? 0.8 : 1), age < 3 ? '#c8ff3d' : '#f3ecd9', prog, alpha);
        });
      }
      if (beat === 'reboot') {
        for (let k = 0; k < 6; k++) {
          const r = ((bt * 0.9 + k * 0.18) % 1.2) * Math.max(w, h);
          ctx.strokeStyle = `rgba(200,255,61,${Math.max(0, 0.6 - r / Math.max(w, h) * 0.5)})`;
          ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(w / 2, skyH * 0.5, r, 0, Math.PI * 2); ctx.stroke();
        }
      }

      // horizon grid
      const gridLight = beat === 'silence' ? 0 : beat === 'boot' || beat === 'flicker' ? (Math.random() < 0.3 ? 0.5 : 0.15) : 1 - corruption * 0.45;
      if (gridLight > 0) drawGrid(t, beat === 'world' ? 'core' : p.grid, beat === 'world' ? 78 : p.hue, corruption, gridLight, still);

      // pulses
      for (let i = pulses.length - 1; i >= 0; i--) {
        const pu = pulses[i];
        const age = (now - pu.born) / 1000;
        if (age > 1.6) { pulses.splice(i, 1); continue; }
        ctx.strokeStyle = pu.color;
        ctx.globalAlpha = Math.max(0, (1 - age / 1.6) * 0.8 * pu.strength);
        ctx.lineWidth = 2 + 6 * (1 - age / 1.6);
        ctx.beginPath(); ctx.arc(pu.x, pu.y, 20 + age * Math.max(w, h) * 0.55, 0, Math.PI * 2); ctx.stroke();
        ctx.globalAlpha = 1;
      }

      // shard particles: burst outward, then home in on the target
      for (let i = particles.length - 1; i >= 0; i--) {
        const q = particles[i];
        if (q.delay > 0) { q.delay -= 1; }
        else {
          const dx = q.tx - q.x, dy = q.ty - q.y;
          const d = Math.hypot(dx, dy);
          q.vx = q.vx * 0.9 + (dx / (d || 1)) * 1.6;
          q.vy = q.vy * 0.9 + (dy / (d || 1)) * 1.6;
          q.x += q.vx; q.y += q.vy;
          if (d < 14) q.life -= 0.2;
        }
        q.life -= 0.004;
        if (q.life <= 0) { particles.splice(i, 1); continue; }
        ctx.fillStyle = q.color;
        ctx.globalAlpha = Math.min(1, q.life * 1.4);
        ctx.fillRect(q.x - q.size / 2, q.y - q.size / 2, q.size, q.size);
      }
      ctx.globalAlpha = 1;

      // corruption tears (skipped under reduced motion)
      const tearAmount = Math.max(tear, corruption * 0.55);
      tear *= 0.92;
      if (!still && tearAmount > 0.05 && Math.random() < tearAmount * 0.5) {
        const bands = 1 + Math.floor(tearAmount * 4);
        for (let b = 0; b < bands; b++) {
          const y = Math.random() * h;
          const bh = 2 + Math.random() * 22 * tearAmount;
          const dx = (Math.random() - 0.5) * 60 * tearAmount;
          try { ctx.drawImage(el, 0, y * dpr, el.width, bh * dpr, dx, y, w, bh); } catch { /* canvas not ready */ }
          ctx.fillStyle = Math.random() < 0.5 ? 'rgba(255,90,78,0.18)' : 'rgba(41,231,255,0.14)';
          ctx.fillRect(0, y, w, Math.max(1, bh * 0.2));
        }
      }
      // data fragments drifting up from the horizon while corrupted
      if (!still && corruption > 0.15) {
        ctx.fillStyle = 'rgba(255,90,78,0.5)';
        const count = Math.floor(corruption * 26);
        for (let i = 0; i < count; i++) {
          const fx0 = (hash(`f${i}`) % 1000) / 1000 * w;
          const fy = skyH - (((t * (12 + (i % 7) * 6) + i * 37) % (skyH * 0.9)));
          ctx.fillRect(fx0, fy, 3 + (i % 3) * 3, 2);
        }
      }
    };
    frame = requestAnimationFrame(loop);
    return () => { cancelAnimationFrame(frame); ro.disconnect(); detach(); };
  }, []);

  return <canvas ref={canvas} className="world-canvas" aria-hidden="true" />;
}
