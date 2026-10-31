/**
 * BOOT: waits for a real user gesture (browsers only allow audio after one),
 * then runs a short boot log built from the actual model the API loaded.
 */
import { useEffect, useRef, useState } from 'react';
import { useGame } from '../game/useGame';
import { audio } from '../audio/engine';
import { music } from '../audio/wire';
import { bus } from '../events/bus';
import { Btn } from '../ui/Button';

export function Boot() {
  const game = useGame();
  const [started, setStarted] = useState(false);
  const [lines, setLines] = useState<{ text: string; tone?: 'ok' | 'warn' | 'bad' }[]>([]);
  const [failed, setFailed] = useState<string | null>(null);
  const once = useRef(false);

  const begin = () => {
    if (once.current) return;
    once.current = true;
    audio.unlock();
    audio.setVolumes({ master: game.settings.masterVolume, music: game.settings.musicVolume, sfx: game.settings.sfxVolume, muted: game.settings.muted });
    setStarted(true);
  };

  useEffect(() => {
    if (started) return;
    const onKey = (e: KeyboardEvent) => { if (!e.metaKey && !e.ctrlKey) begin(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  useEffect(() => {
    if (!started) return;
    let cancelled = false;
    const wait = (ms: number) => new Promise(r => setTimeout(r, game.reducedMotion ? ms / 4 : ms));
    const push = (text: string, tone?: 'ok' | 'warn' | 'bad') => { if (!cancelled) setLines(l => [...l, { text, tone }]); };
    (async () => {
      bus.emit('boot:start', {});
      const pending = game.loadModel();
      for (const [i, text] of ['power bus .......... stable', 'visor ............... cracked', 'antenna ............. bent, receiving'].entries()) {
        await wait(330); bus.emit('boot:relay', { index: i }); push(text, i === 1 ? 'warn' : undefined);
      }
      await wait(300);
      const info = await pending;
      bus.emit('boot:relay', { index: 9 });
      if (!info) {
        push('recognition core .... not responding', 'bad');
        setFailed('The recognition core is unreachable.');
        return;
      }
      push(`recognition core .... ${info.architecture ?? 'cnn'} · ${info.parameters?.toLocaleString() ?? '?'} weights · ${info.classes.length} shapes`, 'ok');
      await wait(260);
      push(`calibration ......... T = ${info.temperature} · unsure below ${Math.round(info.uncertainBelow * 100)}%`, 'ok');
      await wait(260);
      push('memory .............. 50,000,000 drawings received · 0 shapes remembered', 'bad');
      await wait(500);
      bus.emit('boot:signature', {});
      music.setScene('title', { corruption: 1, intensity: 0 });
      await wait(1500);
      if (!cancelled) game.send('BOOT_DONE');
    })();
    return () => { cancelled = true; };
  }, [started]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!started) {
    return (
      <main className="screen boot is-waiting" onPointerDown={begin}>
        <p className="boot-incoming">Incoming transmission</p>
        <button className="boot-receive" onClick={begin} autoFocus>Press any key to receive the signal</button>
        <p className="boot-note">Sound on, if you can. Everything you hear is synthesised live.</p>
      </main>
    );
  }

  return (
    <main className="screen boot">
      <pre className="boot-log" aria-live="polite">
        {lines.map((l, i) => <span key={i} className={l.tone ? `is-${l.tone}` : ''}>{l.text}{'\n'}</span>)}
      </pre>
      {failed && (
        <div className="boot-fail">
          <p>{game.modelError ?? failed}</p>
          <p className="boot-help">Start the backend from the project folder with <code>uvicorn backend.main:app --port 8001</code> and open <code>http://127.0.0.1:8001</code>.</p>
          <div className="row">
            <Btn variant="primary" onClick={async () => { setFailed(null); setLines(l => [...l, { text: 'retrying recognition core…' }]);
              const info = await game.loadModel(); if (info) game.send('BOOT_DONE'); else setFailed('Still unreachable.'); }}>Try again</Btn>
            <Btn onClick={() => { music.setScene('title', { corruption: 1 }); game.send('BOOT_DONE'); }}>Continue without it</Btn>
          </div>
        </div>
      )}
    </main>
  );
}
