/**
 * The drawing challenge. Composition (desktop): the target word set huge on
 * the left with GLITCH and its speech below it; the canvas on the right;
 * controls under the canvas; the model's readout appears after each attempt.
 */
import { useEffect, useRef, useState } from 'react';
import { useGame } from '../game/useGame';
import { DrawingCanvas, type DrawingCanvasHandle } from '../canvas/DrawingCanvas';
import { Glitch } from '../robot/Glitch';
import { Speech } from '../ui/Speech';
import { TargetWord } from '../ui/TargetWord';
import { ScanReadout } from '../ui/ScanReadout';
import { Btn } from '../ui/Button';
import { IconClear, IconEcho, IconSkip, IconTransmit, IconUndo } from '../ui/Icons';
import { contentFor } from '../story/memories';
import { fx } from '../fx/fxController';
import { music } from '../audio/wire';
import { ZONE_MUSIC } from '../audio/music';
import { bus } from '../events/bus';
import type { Mood } from '../robot/moods';
import { Restoration } from './Restoration';

export function Play() {
  const game = useGame();
  const canvas = useRef<DrawingCanvasHandle>(null);
  const robotRef = useRef<HTMLDivElement>(null);
  const [strokeCount, setStrokeCount] = useState(0);
  const [look, setLook] = useState<{ x: number; y: number } | null>(null);
  const [transientMood, setTransientMood] = useState<Mood | null>(null);
  const { phase, session, progress: p, mode } = game;
  const target = session.target;
  const zone = game.currentZone;
  const pending = phase === 'PREDICTION_PENDING';
  const recognized = phase === 'PREDICTION_RECOGNIZED' || phase === 'MEMORY_RESTORATION';
  const missed = phase === 'PREDICTION_MISSED';
  const uncertain = phase === 'PREDICTION_UNCERTAIN';
  const errored = phase === 'PREDICTION_ERROR';
  const content = target ? contentFor(target) : null;

  // music follows the zone and corruption
  useEffect(() => {
    const m = mode === 'campaign' ? ZONE_MUSIC[zone.id] : ZONE_MUSIC['signal-break'];
    music.setScene('zone', { corruption: mode === 'campaign' ? 1 - game.integrity : 0.2, ...(m ?? {}) });
  }, [zone.id, mode, game.integrity]);

  // fresh canvas for every new target
  useEffect(() => { canvas.current?.clear(); setStrokeCount(0); }, [target]);
  useEffect(() => { if (phase === 'DRAWING_READY') { canvas.current?.clear(); setStrokeCount(0); } }, [phase]);

  // recognition: burst the strokes into shards flying into GLITCH's core
  useEffect(() => {
    if (phase !== 'PREDICTION_RECOGNIZED') return;
    setTransientMood('recognising');
    const t = setTimeout(() => setTransientMood('excited'), 600);
    const pts = canvas.current?.screenPoints() ?? [];
    const r = robotRef.current?.getBoundingClientRect();
    if (r && pts.length) {
      const core: [number, number] = [r.left + r.width * 0.5, r.top + r.height * 0.69];
      fx.shards(pts, core, '#c8ff3d');
      fx.pulse(core[0], core[1], '#c8ff3d', 1);
      for (let i = 0; i < 6; i++) setTimeout(() => bus.emit('memory:shard', { index: i }), 500 + i * 90);
    }
    return () => clearTimeout(t);
  }, [phase]);
  useEffect(() => {
    if (missed || uncertain) { fx.glitch(uncertain ? 0.3 : 0.55); setTransientMood(null); }
    if (phase === 'DRAWING_READY') setTransientMood(null);
  }, [phase]); // eslint-disable-line react-hooks/exhaustive-deps

  const submit = () => { if (canvas.current) game.submitDrawing(canvas.current.strokes()); };
  const clear = () => { canvas.current?.clear(); setStrokeCount(0); game.clearDrawing(); };
  const undo = () => {
    if (pending) return;
    canvas.current?.undo();
    if ((canvas.current?.strokes().length ?? 0) === 0) game.clearDrawing();
  };

  // keyboard: Enter transmits, Backspace / Ctrl+Z undoes, Delete clears
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLButtonElement && e.key === 'Enter') return;
      if (e.key === 'Enter' && strokeCount > 0 && !pending) { e.preventDefault(); submit(); }
      else if ((e.key === 'z' && (e.ctrlKey || e.metaKey)) || e.key === 'Backspace') { e.preventDefault(); undo(); }
      else if (e.key === 'Delete') { e.preventDefault(); clear(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const mood: Mood = transientMood ?? (
    pending ? 'scanning'
      : phase === 'DRAWING_IN_PROGRESS' ? 'watching'
      : missed ? (session.attempts >= 3 ? 'frustrated' : 'confused')
      : uncertain ? 'thinking'
      : errored ? 'frustrated'
      : (game.line?.mood ?? 'idle'));

  const tone = pending ? 'scanning' : recognized ? 'recognized' : missed ? 'missed' : uncertain ? 'uncertain' : 'idle';
  const showEcho = session.attempts >= 1 && session.hintLevel < 2 && !recognized;
  const showSkip = mode !== 'rush' ? session.attempts >= 2 && !recognized : !recognized && !pending;
  const recoveredInZone = zone.final ? p.finalRecovered.length : zone.classes.filter(c => p.recovered[c]).length;
  const ticks = game.campaign.classes.map(c => !!p.recovered[c]);
  const canSubmit = strokeCount > 0 && !pending && !recognized;

  if (!target) {
    return (
      <main className="screen play">
        <p className="play-empty">Nothing left to draw here. <Btn onClick={game.toTitle}>Back to the title</Btn></p>
      </main>
    );
  }

  return (
    <main className={`screen play phase-${phase.toLowerCase()} mode-${mode}`}>
      <section className="play-left">
        <p className="play-context">
          {mode === 'campaign' && <>{zone.name} · {recoveredInZone} of {zone.required} to open the gate</>}
          {mode === 'practice' && <>Free drawing · nothing is scored or saved</>}
          {mode === 'revisit' && <>Revisiting dark memories</>}
          {mode === 'rush' && <>Signal rush · {session.rushRecovered.length} recognised · {session.rushScore} points</>}
        </p>
        <TargetWord word={target} state={recognized ? 'restored' : pending ? 'scanning' : 'lost'} />
        {mode === 'rush' && game.rushLeft !== null && (
          <div className="rush-timer" aria-live="off" style={{ ['--left' as any]: game.rushLeft / 75 }}>{game.rushLeft}s</div>
        )}
        <div className="play-glitch">
          <div className="play-robot" ref={robotRef}>
            <Glitch mood={mood} integrity={mode === 'campaign' ? game.integrity : 0.9} ticks={ticks} rings={p.zonesCompleted.length}
              totalRings={game.campaign.zones.length} lookAt={phase === 'DRAWING_IN_PROGRESS' ? look : null} reducedMotion={game.reducedMotion} />
          </div>
          {game.line && <Speech text={game.line.text} speed={game.settings.textSpeed} lineKey={game.line.key} className="play-speech" />}
        </div>
        {session.hintLevel > 0 && content && (
          <ol className="echoes" aria-label="Echoes">
            {content.echoes.slice(0, session.hintLevel).map((e, i) => <li key={i}>{e}</li>)}
          </ol>
        )}
        {session.result && game.settings.showModelDetails && !pending && (
          <ScanReadout result={session.result} target={target} compact={recognized} />
        )}
      </section>

      <section className="play-right">
        <div className={`canvas-frame tone-${tone}`}>
          <svg className="frame-edge" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
            <polygon points="3.5,0 100,0 100,96.5 96.5,100 0,100 0,3.5" vectorEffect="non-scaling-stroke" />
          </svg>
          <DrawingCanvas ref={canvas} locked={pending || recognized} tone={tone}
            onStrokeStart={game.strokeStarted} onChange={setStrokeCount} onPointer={setLook}
            ariaLabel={`Drawing canvas. Draw ${target} in your own way, then transmit.`} />
          {strokeCount === 0 && phase === 'DRAWING_READY' && <p className="canvas-prompt">Draw {target} here, your way</p>}
          {pending && <p className="canvas-status" role="status">Scanning…</p>}
        </div>
        <div className="play-controls">
          <Btn variant="icon" onClick={undo} disabled={pending || strokeCount === 0 || recognized} title="Undo (Backspace)" aria-label="Undo last stroke"><IconUndo /></Btn>
          <Btn variant="icon" onClick={clear} disabled={strokeCount === 0 || recognized} title="Clear (Delete)" aria-label="Clear the canvas"><IconClear /></Btn>
          <span className="spacer" />
          {showEcho && (
            <Btn variant="ghost" onClick={game.revealHint} title="Use an echo: a hint in words. Costs a little score.">
              <IconEcho /> {session.hintLevel === 0 ? 'Ask for an echo' : 'A clearer echo'}
            </Btn>
          )}
          {showSkip && <Btn variant="quiet" sound="back" onClick={game.skipTarget} title="Leave this memory for later"><IconSkip /> Later</Btn>}
          <Btn variant="primary" sound="confirm" onClick={submit} disabled={!canSubmit} className="transmit" aria-keyshortcuts="Enter">
            <IconTransmit /> {errored ? 'Send again' : missed || uncertain ? 'Transmit again' : 'Transmit'}
          </Btn>
        </div>
        {errored && session.error && <p className="play-error" role="alert">{session.error.message}</p>}
      </section>

      {phase === 'MEMORY_RESTORATION' && game.restoration && <Restoration />}
      {mode === 'rush' && game.rushOver && (
        <div className="rush-over" role="dialog" aria-modal="true" aria-label="Signal rush finished">
          <h2>Signal rush over</h2>
          <p className="big-number">{session.rushScore}</p>
          <p>{session.rushRecovered.length} drawings recognised{session.rushScore >= p.rushBest && session.rushScore > 0 ? ' · new best' : ` · best ${p.rushBest}`}</p>
          <Btn variant="primary" sound="confirm" autoFocus onClick={game.toTitle}>Back to the title</Btn>
        </div>
      )}
    </main>
  );
}
