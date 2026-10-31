/**
 * MEMORY_CORE_ACTIVATION: the player presses and holds GLITCH's core (pointer
 * or Space). Misaligned rings rotate into place as charge builds; letting go
 * drains it. At full charge the core comes online and GLITCH explains the rules.
 */
import { useEffect, useRef, useState } from 'react';
import { useGame } from '../game/useGame';
import { Cinematic } from './Cinematic';
import { Speech } from '../ui/Speech';
import { bus } from '../events/bus';
import { fx } from '../fx/fxController';
import { CORE_ACTIVATION, CORE_ONLINE } from '../story/dialogue';

export function CoreActivation() {
  const game = useGame();
  const [charge, setCharge] = useState(0);
  const [online, setOnline] = useState(false);
  const holding = useRef(false);
  const chargeRef = useRef(0);
  const coreEl = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (online) return;
    let frame = 0, last = performance.now(), lastTick = 0;
    const loop = (now: number) => {
      const dt = (now - last) / 1000; last = now;
      const rate = game.reducedMotion ? 0.9 : 0.48;
      chargeRef.current = Math.max(0, Math.min(1, chargeRef.current + (holding.current ? rate : -0.6) * dt));
      setCharge(chargeRef.current);
      if (holding.current && now - lastTick > 110) { lastTick = now; bus.emit('core:charge', { level: chargeRef.current }); }
      if (chargeRef.current >= 1) {
        setOnline(true);
        bus.emit('core:online', {});
        const r = coreEl.current?.getBoundingClientRect();
        if (r) fx.pulse(r.left + r.width / 2, r.top + r.height / 2, '#c8ff3d', 1.4);
        return;
      }
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    const down = (e: KeyboardEvent) => { if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); holding.current = true; } };
    const up = (e: KeyboardEvent) => { if (e.key === ' ' || e.key === 'Enter') holding.current = false; };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => { cancelAnimationFrame(frame); window.removeEventListener('keydown', down); window.removeEventListener('keyup', up); };
  }, [online, game.reducedMotion]);

  if (online) {
    return (
      <Cinematic beats={CORE_ONLINE} speed={game.settings.textSpeed} integrity={0.05} ticks={game.campaign.classes.map(() => false)}
        rings={0} totalRings={game.campaign.zones.length} reducedMotion={game.reducedMotion}
        onDone={() => { game.markIntroSeen(); game.send('CORE_ONLINE'); }} className="core-online" />
    );
  }

  const rings = [0, 1, 2, 3];
  return (
    <main className="screen core-activation">
      <Speech text={CORE_ACTIVATION[0].text} speed={game.settings.textSpeed} lineKey="core" className="core-speech" />
      <button ref={coreEl} className="core-button" aria-label="Press and hold to power GLITCH's memory core"
        onPointerDown={e => { e.preventDefault(); (e.target as Element).setPointerCapture?.(e.pointerId); holding.current = true; }}
        onPointerUp={() => (holding.current = false)} onPointerCancel={() => (holding.current = false)} onPointerLeave={() => (holding.current = false)}
        style={{ ['--charge' as any]: charge }}>
        <svg viewBox="0 0 300 300" aria-hidden="true">
          {rings.map(i => {
            const r = 60 + i * 26;
            const misalign = (1 - charge) * (i % 2 ? -1 : 1) * (40 + i * 25);
            return (
              <circle key={i} cx="150" cy="150" r={r} fill="none" strokeWidth={i === 0 ? 4 : 2}
                stroke={charge > 0.98 ? 'var(--lime)' : i <= charge * 4 ? 'var(--cyan)' : 'rgba(243,236,217,.25)'}
                strokeDasharray={`${(r * 2 * Math.PI) / (8 + i * 4) * (0.35 + 0.6 * charge)} ${(r * 2 * Math.PI) / (8 + i * 4) * (0.65 - 0.6 * charge) + 0.01}`}
                transform={`rotate(${misalign} 150 150)`} />
            );
          })}
          <circle cx="150" cy="150" r={18 + 30 * charge} fill="var(--lime)" opacity={0.15 + 0.6 * charge} />
          <circle cx="150" cy="150" r="10" fill="var(--ivory)" opacity={0.4 + 0.6 * charge} />
        </svg>
        <span className="core-label">{charge > 0.02 ? `${Math.round(charge * 100)}%` : 'Press and hold'}</span>
      </button>
      <p className="core-hint">Hold with the mouse, a finger, or the space bar.</p>
    </main>
  );
}
