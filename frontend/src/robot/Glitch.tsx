/**
 * GLITCH, drawn in SVG. The silhouette: a CRT-lantern head with a dark visor,
 * a bent antenna ending in a receiver dish, a tilted capsule body with a round
 * memory-core window, one stubby arm, one floating hand, and a hover thruster.
 *
 * The memory core is real game state: one tick per class the CNN knows (lit
 * when that memory is restored) and one ring per completed zone.
 *
 * Pose changes are declarative (POSES[mood]); the continuous motion (bob,
 * blink, antenna sway, gaze, sparks) runs in one requestAnimationFrame loop
 * that writes transforms straight to the DOM.
 */
import { useEffect, useMemo, useRef } from 'react';
import { POSES, type Mood } from './moods';

export interface GlitchProps {
  mood: Mood;
  /** 0 = freshly corrupted, 1 = fully restored. */
  integrity: number;
  ticks?: boolean[];
  rings?: number;
  totalRings?: number;
  /** Where to look, in -1..1 coordinates relative to the robot. */
  lookAt?: { x: number; y: number } | null;
  reducedMotion?: boolean;
  className?: string;
  label?: string;
}

const GLOW: Record<string, string> = {
  cyan: 'var(--cyan)', lime: 'var(--lime)', coral: 'var(--coral)', violet: 'var(--violet)', ivory: 'var(--ivory)',
};

function Eye({ kind, cx, cy, s, side }: { kind: string; cx: number; cy: number; s: number; side: 1 | -1 }) {
  const c = 'var(--robot-glow)';
  switch (kind) {
    case 'off': return <line x1={cx - 9} y1={cy} x2={cx + 9} y2={cy} stroke="var(--robot-dim)" strokeWidth={3} strokeLinecap="round" />;
    case 'happy': return <path d={`M ${cx - 10 * s} ${cy + 4} Q ${cx} ${cy - 10 * s} ${cx + 10 * s} ${cy + 4}`} fill="none" stroke={c} strokeWidth={4} strokeLinecap="round" />;
    case 'squint': return <path d={`M ${cx - 10} ${cy + side * 2} L ${cx + 10} ${cy - side * 2}`} stroke={c} strokeWidth={4} strokeLinecap="round" />;
    case 'question': return side === 1
      ? <path d={`M ${cx - 6} ${cy - 7} Q ${cx} ${cy - 15} ${cx + 6} ${cy - 7} Q ${cx + 8} ${cy - 1} ${cx} ${cy + 2} L ${cx} ${cy + 5} M ${cx} ${cy + 11} L ${cx} ${cy + 11.5}`} fill="none" stroke={c} strokeWidth={3.5} strokeLinecap="round" />
      : <rect x={cx - 8 * s} y={cy - 11 * s} width={16 * s} height={22 * s} rx={7 * s} fill={c} />;
    case 'scan': return <rect x={cx - 11} y={cy - 2.5} width={22} height={5} rx={2.5} fill={c} className="eye-scan" />;
    case 'heart': return <path d={`M ${cx} ${cy + 9} C ${cx - 14} ${cy} ${cx - 9} ${cy - 11} ${cx} ${cy - 4} C ${cx + 9} ${cy - 11} ${cx + 14} ${cy} ${cx} ${cy + 9} Z`} fill={c} />;
    case 'wide': return <rect x={cx - 10 * s} y={cy - 13 * s} width={20 * s} height={26 * s} rx={9 * s} fill={c} />;
    default: return <rect x={cx - 8 * s} y={cy - 11 * s} width={16 * s} height={22 * s} rx={7 * s} fill={c} />;
  }
}

function Mouth({ kind }: { kind: string }) {
  const c = 'var(--robot-glow)';
  const y = 159;
  switch (kind) {
    case 'smile': return <path d={`M 108 ${y - 2} Q 120 ${y + 7} 132 ${y - 2}`} fill="none" stroke={c} strokeWidth={3} strokeLinecap="round" />;
    case 'grin': return <path d={`M 104 ${y - 3} Q 120 ${y + 11} 136 ${y - 3} Z`} fill={c} opacity={0.9} />;
    case 'wobble': return <path d={`M 106 ${y} q 4.5 -4 9 0 t 9 0 t 9 0`} fill="none" stroke={c} strokeWidth={2.6} strokeLinecap="round" />;
    case 'o': return <circle cx={120} cy={y} r={3.6} fill="none" stroke={c} strokeWidth={2.4} />;
    case 'dots': return <g fill={c} className="mouth-dots"><circle cx={110} cy={y} r={2.2} /><circle cx={120} cy={y} r={2.2} /><circle cx={130} cy={y} r={2.2} /></g>;
    case 'flat': return <line x1={111} y1={y} x2={129} y2={y} stroke={c} strokeWidth={2.6} strokeLinecap="round" />;
    default: return null;
  }
}

export function Glitch({ mood, integrity, ticks = [], rings = 0, totalRings = 6, lookAt = null, reducedMotion = false, className = '', label }: GlitchProps) {
  const pose = POSES[mood];
  const root = useRef<SVGSVGElement>(null);
  const body = useRef<SVGGElement>(null);
  const antenna = useRef<SVGGElement>(null);
  const eyes = useRef<SVGGElement>(null);
  const lids = useRef<SVGRectElement>(null);
  const hand = useRef<SVGGElement>(null);
  const flame = useRef<SVGPathElement>(null);
  const spark = useRef<SVGGElement>(null);
  const live = useRef({ pose, lookAt, integrity, reducedMotion });
  live.current = { pose, lookAt, integrity, reducedMotion };

  useEffect(() => {
    let frame = 0;
    let nextBlink = performance.now() + 1800;
    let blinkUntil = 0;
    let gx = 0, gy = 0;
    const loop = (now: number) => {
      const { pose: p, lookAt: look, integrity: health, reducedMotion: still } = live.current;
      const t = now / 1000;
      const bob = still ? 0 : Math.sin(t * 2.1) * p.bob;
      const shake = !still && p.shake ? Math.sin(t * 48) * 1.6 : 0;
      body.current?.setAttribute('transform', `translate(${shake} ${bob}) rotate(${p.tilt} 120 190)`);
      // the antenna droops when damaged and straightens as integrity returns
      const droop = (1 - health) * 26;
      const sway = still ? 0 : Math.sin(t * 1.7 * p.antennaSpeed) * (6 + 6 * (1 - health));
      antenna.current?.setAttribute('transform', `rotate(${sway + droop} 138 84)`);
      // gaze eases toward the pointer
      const tx = look ? Math.max(-1, Math.min(1, look.x)) * 6 : 0;
      const ty = look ? Math.max(-1, Math.min(1, look.y)) * 4 : 0;
      gx += (tx - gx) * 0.12; gy += (ty - gy) * 0.12;
      eyes.current?.setAttribute('transform', `translate(${gx.toFixed(2)} ${gy.toFixed(2)})`);
      if (!still && now > nextBlink) { blinkUntil = now + 120; nextBlink = now + 2200 + Math.random() * 3200; }
      lids.current?.setAttribute('height', now < blinkUntil && p.eyes !== 'off' ? '40' : '0');
      if (hand.current) {
        const wave = p.eyes === 'happy' && !still ? Math.sin(t * 9) * 10 : 0;
        hand.current.setAttribute('transform', `translate(0 ${still ? 0 : Math.sin(t * 2.1 + 1.3) * 5}) rotate(${wave} 196 214)`);
      }
      if (flame.current && !still) flame.current.setAttribute('transform', `scale(1 ${0.8 + Math.random() * 0.35})`);
      if (spark.current) spark.current.style.opacity = !still && health < 0.35 && Math.random() < 0.025 ? '1' : '0';
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, []);

  const count = ticks.length || 32;
  const tickMarks = useMemo(() => Array.from({ length: count }, (_, i) => {
    const a = (i / count) * Math.PI * 2 - Math.PI / 2;
    const r0 = 30, r1 = 36;
    return { x0: 120 + Math.cos(a) * r0, y0: 222 + Math.sin(a) * r0, x1: 120 + Math.cos(a) * r1, y1: 222 + Math.sin(a) * r1 };
  }), [count]);

  const crack = integrity < 0.32;
  const patched = integrity < 0.6;
  const style = { ['--robot-glow' as any]: GLOW[pose.glow], ['--robot-dim' as any]: 'rgba(243,236,217,.35)' };

  return (
    <svg ref={root} viewBox="0 0 240 320" className={`glitch-robot mood-${mood} ${className}`} style={style}
      role="img" aria-label={label ?? `GLITCH looks ${mood}`}>
      <defs>
        <radialGradient id="g-core" cx="50%" cy="50%" r="50%">
          <stop offset="0" stopColor="var(--robot-glow)" stopOpacity="0.95" />
          <stop offset="0.55" stopColor="var(--robot-glow)" stopOpacity="0.25" />
          <stop offset="1" stopColor="var(--robot-glow)" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="g-shell" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#2a2f45" />
          <stop offset="1" stopColor="#141726" />
        </linearGradient>
        <filter id="g-bloom" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      {/* ground shadow */}
      <ellipse cx="120" cy="306" rx={44 - pose.bob} ry="6" fill="#000" opacity="0.45" />

      <g ref={body}>
        {/* thruster */}
        <path d="M 108 262 L 132 262 L 126 276 L 114 276 Z" fill="#1b1f30" stroke="var(--line)" strokeWidth="2" />
        <g style={{ transformOrigin: '120px 276px' }}>
          <path ref={flame} d="M 113 277 Q 120 304 127 277 Z" fill="var(--robot-glow)" opacity="0.55" filter="url(#g-bloom)" style={{ transformOrigin: '120px 277px' }} />
        </g>

        {/* antenna: bent mast + receiver dish */}
        <g ref={antenna}>
          <path d="M 138 84 L 146 44 L 170 28" fill="none" stroke="var(--line)" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M 162 16 A 13 13 0 0 1 182 38" fill="none" stroke="var(--robot-glow)" strokeWidth="4" strokeLinecap="round" filter="url(#g-bloom)" />
          <circle cx="170" cy="28" r="4" fill="var(--robot-glow)" className={integrity < 0.5 ? 'antenna-flicker' : ''} />
        </g>

        {/* arm (left) */}
        <path d="M 72 212 Q 52 220 50 244" fill="none" stroke="var(--line)" strokeWidth="7" strokeLinecap="round" />
        <circle cx="50" cy="248" r="9" fill="url(#g-shell)" stroke="var(--line)" strokeWidth="2.5" />

        {/* body capsule */}
        <path d="M 74 190 Q 70 176 86 172 L 154 172 Q 170 176 166 190 L 160 252 Q 158 266 142 266 L 98 266 Q 82 266 80 252 Z"
          fill="url(#g-shell)" stroke="var(--line)" strokeWidth="3" strokeLinejoin="round" />
        {patched && <g className="patch"><rect x="138" y="236" width="20" height="9" rx="2" transform="rotate(-18 148 240)" fill="var(--coral)" opacity="0.85" />
          <line x1="140" y1="242" x2="156" y2="236" stroke="#06070b" strokeWidth="1.5" opacity="0.5" /></g>}

        {/* memory core window: one tick per class, one ring per zone */}
        <circle cx="120" cy="222" r="38" fill="#07080d" stroke="var(--line)" strokeWidth="2.5" />
        <circle cx="120" cy="222" r={10 + 16 * integrity} fill="url(#g-core)" className="core-glow" />
        {Array.from({ length: totalRings }, (_, i) => (
          <circle key={i} cx="120" cy="222" r={6 + i * 4} fill="none" strokeWidth="1.4"
            stroke={i < rings ? 'var(--lime)' : 'rgba(243,236,217,0.12)'} strokeDasharray={i < rings ? undefined : '2 3'} />
        ))}
        {tickMarks.map((m, i) => (
          <line key={i} x1={m.x0} y1={m.y0} x2={m.x1} y2={m.y1} strokeWidth="2.4" strokeLinecap="round"
            stroke={ticks[i] ? 'var(--lime)' : 'rgba(243,236,217,0.16)'} />
        ))}

        {/* neck */}
        <path d="M 108 164 L 132 164 M 110 169 L 130 169" stroke="var(--line)" strokeWidth="3" strokeLinecap="round" />

        {/* head: CRT lantern */}
        <path d="M 58 98 Q 58 82 74 80 L 166 80 Q 182 82 182 98 L 178 146 Q 176 162 160 162 L 80 162 Q 64 162 62 146 Z"
          fill="url(#g-shell)" stroke="var(--line)" strokeWidth="3" strokeLinejoin="round" />
        <circle cx="56" cy="120" r="7" fill="#141726" stroke="var(--line)" strokeWidth="2.5" />
        <circle cx="184" cy="120" r="7" fill="#141726" stroke="var(--line)" strokeWidth="2.5" />
        {/* visor */}
        <path d="M 74 98 Q 74 92 80 92 L 160 92 Q 166 92 166 98 L 163 140 Q 162 150 152 150 L 88 150 Q 78 150 77 140 Z" fill="#05060a" />
        <path d="M 82 97 L 112 97" stroke="rgba(243,236,217,.18)" strokeWidth="3" strokeLinecap="round" />
        <g ref={eyes} filter="url(#g-bloom)">
          <Eye kind={pose.eyes} cx={100} cy={122} s={pose.eyeScale} side={-1} />
          <Eye kind={pose.eyes} cx={140} cy={122} s={pose.eyeScale} side={1} />
          <Mouth kind={pose.mouth} />
        </g>
        <rect ref={lids} x="74" y="100" width="92" height="0" fill="#05060a" />
        {crack && <path d="M 150 93 L 141 108 L 147 114 L 136 131" fill="none" stroke="rgba(243,236,217,.55)" strokeWidth="1.4" className="visor-crack" />}

        {/* floating hand (right) */}
        <g ref={hand}>
          <circle cx="198" cy="212" r="11" fill="url(#g-shell)" stroke="var(--line)" strokeWidth="2.5" />
          <path d="M 192 206 L 192 200 M 198 204 L 198 197 M 204 206 L 204 200" stroke="var(--line)" strokeWidth="3" strokeLinecap="round" />
          <circle cx="182" cy="214" r="2" fill="var(--robot-glow)" opacity="0.7" />
          <circle cx="175" cy="215" r="1.4" fill="var(--robot-glow)" opacity="0.4" />
        </g>

        <g ref={spark} style={{ opacity: 0 }}>
          <path d="M 160 120 l 8 -6 l -2 6 l 8 -4" fill="none" stroke="var(--coral)" strokeWidth="2" />
        </g>
      </g>
    </svg>
  );
}
