import { useEffect, useRef, useState, type ReactNode } from 'react';
import { useGame } from '../game/useGame';
import { useOverlay } from '../ui/overlay';
import { Btn } from '../ui/Button';
import { IconClose } from '../ui/Icons';
import { StrokeArt } from '../ui/StrokeArt';
import { contentFor } from '../story/memories';
import { ACHIEVEMENTS } from '../game/achievements';
import { rankFor } from '../game/scoring';

function Panel({ title, children, className = '' }: { title: string; children: ReactNode; className?: string }) {
  const { open } = useOverlay();
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    ref.current?.querySelector<HTMLElement>('button, input, select')?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); open(null); } };
    window.addEventListener('keydown', onKey, true);
    return () => { window.removeEventListener('keydown', onKey, true); prev?.focus?.(); };
  }, [open]);
  return (
    <div className="overlay" role="dialog" aria-modal="true" aria-label={title} onPointerDown={e => { if (e.target === e.currentTarget) open(null); }}>
      <div ref={ref} className={`panel ${className}`}>
        <header className="panel-head">
          <h2>{title}</h2>
          <Btn variant="icon" sound="back" onClick={() => open(null)} aria-label="Close"><IconClose /></Btn>
        </header>
        <div className="panel-body">{children}</div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ archive
export function Archive() {
  const game = useGame();
  const { open } = useOverlay();
  const p = game.progress;
  const [selected, setSelected] = useState<string | null>(null);
  const [tab, setTab] = useState<'memories' | 'achievements'>('memories');
  const zones = game.campaign.zones.filter(z => !z.final);
  const sel = selected ? p.recovered[selected] : null;
  const zoneReached = (i: number) => p.zonesCompleted.length >= i || p.campaignComplete;
  const darkReachable = zones.flatMap((z, i) => (zoneReached(i + 1) ? z.classes.filter(c => !p.recovered[c]) : []));
  const total = game.campaign.classes.length;
  const count = Object.keys(p.recovered).length;
  return (
    <Panel title="Memory archive" className="archive">
      <div className="archive-tabs" role="tablist">
        <button role="tab" aria-selected={tab === 'memories'} onClick={() => setTab('memories')}>Memories {count}/{total}</button>
        <button role="tab" aria-selected={tab === 'achievements'} onClick={() => setTab('achievements')}>Achievements {p.achievements.length}/{ACHIEVEMENTS.length}</button>
      </div>
      {tab === 'memories' ? (
        <div className="archive-layout">
          <div className="archive-zones">
            {zones.map(z => (
              <section key={z.id} className="archive-zone">
                <h3>{z.name}</h3>
                <ul>
                  {z.classes.map(c => {
                    const m = p.recovered[c];
                    return (
                      <li key={c}>
                        <button className={`archive-cell${m ? ' is-lit' : ''}${selected === c ? ' is-selected' : ''}`} onClick={() => setSelected(c)}
                          aria-label={m ? `${c}, restored` : `${c}, still dark`}>
                          {m ? <StrokeArt strokes={m.strokes} /> : <span className="dark-glyph" aria-hidden="true">{p.seen.includes(c) ? c[0] : '?'}</span>}
                          <span className="cell-name">{m || p.seen.includes(c) ? c : '· · ·'}</span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </section>
            ))}
          </div>
          <aside className="archive-detail" aria-live="polite">
            {selected && sel ? (
              <>
                <StrokeArt strokes={sel.strokes} label={`Your drawing of ${selected}`} className="detail-art" />
                <h3>{selected}</h3>
                <p className="detail-memory">{contentFor(selected).memory}</p>
                <dl className="detail-stats">
                  <div><dt>Attempts</dt><dd>{sel.attempts}</dd></div>
                  <div><dt>Echoes</dt><dd>{sel.hintsUsed}</dd></div>
                  <div><dt>Signal clarity</dt><dd>{Math.round(sel.clarity * 100)}%</dd></div>
                  <div><dt>Points</dt><dd>{sel.points}</dd></div>
                </dl>
                <p className="detail-meta">Recognised by {sel.modelVersion} on {new Date(sel.at).toLocaleDateString()}.</p>
              </>
            ) : selected ? (
              <>
                <h3>{p.seen.includes(selected) ? selected : 'Unknown memory'}</h3>
                <p className="detail-memory">This memory is still dark.</p>
                {darkReachable.includes(selected) && game.model && (
                  <Btn variant="primary" sound="confirm" onClick={() => { open(null); game.startRevisit([selected, ...darkReachable.filter(c => c !== selected)]); }}>
                    Try to recover it
                  </Btn>
                )}
              </>
            ) : (
              <p className="detail-empty">Every memory you restore is kept here as you drew it. Select one to see it.</p>
            )}
            {!selected && darkReachable.length > 0 && game.model && (
              <Btn onClick={() => { open(null); game.startRevisit(darkReachable); }}>Revisit all {darkReachable.length} dark memories</Btn>
            )}
          </aside>
        </div>
      ) : (
        <ul className="achievement-list">
          {ACHIEVEMENTS.map(a => (
            <li key={a.id} className={p.achievements.includes(a.id) ? 'is-unlocked' : ''}>
              <strong>{a.name}</strong><span>{a.description}</span>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

// ----------------------------------------------------------------- settings
function Slider({ label, value, onChange, disabled }: { label: string; value: number; onChange(v: number): void; disabled?: boolean }) {
  return (
    <label className="setting">
      <span>{label}</span>
      <input type="range" min={0} max={100} value={Math.round(value * 100)} disabled={disabled} onChange={e => onChange(Number(e.target.value) / 100)} />
      <output>{Math.round(value * 100)}</output>
    </label>
  );
}

export function Settings() {
  const game = useGame();
  const s = game.settings;
  const [confirmReset, setConfirmReset] = useState(false);
  return (
    <Panel title="Settings" className="settings">
      <fieldset>
        <legend>Sound</legend>
        <label className="setting toggle"><span>Mute everything</span><input type="checkbox" checked={s.muted} onChange={e => game.updateSettings({ muted: e.target.checked })} /></label>
        <Slider label="Master" value={s.masterVolume} disabled={s.muted} onChange={v => game.updateSettings({ masterVolume: v })} />
        <Slider label="Music" value={s.musicVolume} disabled={s.muted} onChange={v => game.updateSettings({ musicVolume: v })} />
        <Slider label="Effects" value={s.sfxVolume} disabled={s.muted} onChange={v => game.updateSettings({ sfxVolume: v })} />
        <label className="setting toggle"><span>Sounds while drawing</span><input type="checkbox" checked={s.drawingSounds} onChange={e => game.updateSettings({ drawingSounds: e.target.checked })} /></label>
      </fieldset>
      <fieldset>
        <legend>Display and reading</legend>
        <label className="setting"><span>Reduced motion</span>
          <select value={s.reducedMotion} onChange={e => game.updateSettings({ reducedMotion: e.target.value as typeof s.reducedMotion })}>
            <option value="system">Follow my system</option><option value="on">On</option><option value="off">Off</option>
          </select></label>
        <label className="setting"><span>Text speed</span>
          <select value={s.textSpeed} onChange={e => game.updateSettings({ textSpeed: e.target.value as typeof s.textSpeed })}>
            <option value="normal">Normal</option><option value="fast">Fast</option><option value="instant">Instant</option>
          </select></label>
        <label className="setting toggle"><span>Show what the network saw after each drawing</span><input type="checkbox" checked={s.showModelDetails} onChange={e => game.updateSettings({ showModelDetails: e.target.checked })} /></label>
      </fieldset>
      <fieldset>
        <legend>Saved progress</legend>
        <p className="setting-note">Progress is saved in this browser only.</p>
        {!confirmReset ? <Btn variant="quiet" onClick={() => setConfirmReset(true)}>Erase saved progress…</Btn> : (
          <div className="confirm-row"><span>This can’t be undone.</span>
            <Btn variant="primary" onClick={() => { game.resetSave(); setConfirmReset(false); }}>Erase it</Btn>
            <Btn variant="quiet" sound="back" onClick={() => setConfirmReset(false)}>Cancel</Btn></div>
        )}
      </fieldset>
    </Panel>
  );
}

// ------------------------------------------------------- inside GLITCH's head
const pct = (v: unknown, d = 1) => (typeof v === 'number' ? `${(v * 100).toFixed(d)}%` : 'n/a');

export function Inside() {
  const game = useGame();
  const m = game.model;
  return (
    <Panel title="Inside GLITCH’s head" className="inside">
      {!m ? <p>The recognition core is offline, so there is nothing to report. Start the backend and reload.</p> : (
        <>
          <p className="inside-lede">GLITCH recognises drawings with a convolutional neural network trained offline on Quick, Draw! sketches. Everything below comes from the model the server loaded.</p>
          <dl className="inside-stats">
            <div><dt>Model version</dt><dd>{m.modelVersion}</dd></div>
            <div><dt>Architecture</dt><dd>{m.architecture}</dd></div>
            <div><dt>Trainable weights</dt><dd>{m.parameters?.toLocaleString()}</dd></div>
            <div><dt>Shapes it knows</dt><dd>{m.classes.length}</dd></div>
            <div><dt>Held-out test accuracy</dt><dd>{pct(m.testMetrics.top_1_accuracy)} <small>top-1</small> · {pct(m.testMetrics.top_3_accuracy)} <small>top-3</small></dd></div>
            <div><dt>Macro F1</dt><dd>{typeof m.testMetrics.macro_f1 === 'number' ? m.testMetrics.macro_f1.toFixed(3) : 'n/a'}</dd></div>
            <div><dt>Calibration error (ECE)</dt><dd>{pct(m.testMetrics.ece_temperature_scaled, 2)} <small>after temperature T = {m.temperature}</small></dd></div>
            <div><dt>“Unsure” below</dt><dd>{Math.round(m.uncertainBelow * 100)}% <small>fitted on validation data</small></dd></div>
            <div><dt>Mean inference this session</dt><dd>{m.meanInferenceMs ? `${m.meanInferenceMs} ms` : 'no drawings yet'}</dd></div>
          </dl>
          <h3>What happens to your drawing</h3>
          <ol className="inside-steps">
            <li>Your strokes are rendered as black ink on a white 600×600 image.</li>
            <li>The server marks every non-white pixel as ink, crops to your drawing and centres it in a square with a 12% margin.</li>
            <li>Strokes are thickened so thin lines survive, then the image is shrunk to 28×28 and inverted to white-on-black, the format of the training data.</li>
            <li>The network outputs a score for each of the {m.classes.length} shapes. Scores are temperature-scaled so they track how often the network is actually right.</li>
            <li>If the top score is below the “unsure” line, GLITCH says it isn’t sure. A memory is restored only if the top shape is the word you were given.</li>
          </ol>
          <p className="inside-note">{m.note}</p>
          <p className="inside-classes">{m.classes.join(' · ')}</p>
          <p className="inside-rank">Your rank: {rankFor(game.progress.xp).name}</p>
        </>
      )}
    </Panel>
  );
}

// --------------------------------------------------------------- pause menu
export function Menu() {
  const game = useGame();
  const { open } = useOverlay();
  return (
    <Panel title="Paused" className="menu">
      <nav className="menu-list">
        <Btn variant="primary" sound="confirm" onClick={() => open(null)}>Resume</Btn>
        <Btn onClick={() => open('archive')}>Memory archive</Btn>
        <Btn onClick={() => open('settings')}>Settings</Btn>
        <Btn onClick={() => open('inside')}>Inside GLITCH’s head</Btn>
        <Btn variant="quiet" sound="back" onClick={() => { open(null); game.toTitle(); }}>Save and return to title</Btn>
      </nav>
    </Panel>
  );
}
