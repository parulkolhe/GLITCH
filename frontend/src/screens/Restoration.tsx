/** MEMORY_RESTORATION: the recovered memory, the player's own drawing, and the score. */
import { useEffect } from 'react';
import { useGame } from '../game/useGame';
import { StrokeArt } from '../ui/StrokeArt';
import { Btn } from '../ui/Button';
import { useTypewriter } from '../ui/Speech';
import { contentFor } from '../story/memories';
import { achievementById } from '../game/achievements';

export function Restoration() {
  const game = useGame();
  const r = game.restoration!;
  const rush = game.mode === 'rush';
  const memory = contentFor(r.target).memory;
  const { shown, done, finish } = useTypewriter(memory, rush ? 'instant' : game.settings.textSpeed, r.target, false);
  const total = game.campaign.classes.length;
  const recovered = Object.keys(game.progress.recovered).length;

  useEffect(() => {
    if (rush) { const id = setTimeout(game.continueAfterRestoration, 900); return () => clearTimeout(id); }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); if (!done) finish(); else game.continueAfterRestoration(); } };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [done, rush]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className={`restoration${rush ? ' is-rush' : ''}`} role="dialog" aria-modal="true" aria-labelledby="restored-title">
      <div className="rest-art">
        <StrokeArt strokes={r.strokes} label={`Your drawing of ${r.target}`} />
      </div>
      <div className="rest-body">
        <p className="rest-kicker">{game.mode === 'practice' || rush ? 'Recognised' : `Memory restored · ${recovered} of ${total}`}</p>
        <h2 id="restored-title" className="rest-word">{r.target}</h2>
        {!rush && <p className="rest-memory" onClick={finish}><span className="sr-only">{memory}</span><span aria-hidden="true">{shown}</span></p>}
        {game.mode !== 'practice' && (
          <dl className="rest-score">
            {r.lines.map(l => <div key={l.label}><dt>{l.label}</dt><dd>{l.value}</dd></div>)}
            <div className="rest-total"><dt>Total</dt><dd>{r.points}</dd></div>
          </dl>
        )}
        {r.achievements.length > 0 && (
          <ul className="rest-achievements">{r.achievements.map(id => <li key={id}>Achievement: {achievementById(id)?.name}</li>)}</ul>
        )}
        {!rush && (
          <Btn variant="primary" sound="confirm" autoFocus onClick={done ? game.continueAfterRestoration : finish}>
            {r.zoneDone ? 'Open the gate' : 'Next memory'}
          </Btn>
        )}
      </div>
    </div>
  );
}
