/** FINAL_SEQUENCE and CAMPAIGN_COMPLETE. */
import { useEffect, useMemo } from 'react';
import { useGame } from '../game/useGame';
import { Cinematic } from './Cinematic';
import { finaleBeats, type FinaleFacts } from '../story/dialogue';
import { topConfusions } from '../game/progress';
import { bus } from '../events/bus';
import { Btn } from '../ui/Button';
import { useOverlay } from '../ui/overlay';
import { ACHIEVEMENTS } from '../game/achievements';
import { rankFor } from '../game/scoring';
import { Wordmark } from '../ui/Wordmark';

export function finaleFacts(game: ReturnType<typeof useGame>): FinaleFacts {
  const p = game.progress;
  const entries = Object.entries(p.recovered);
  const clearest = entries.sort((a, b) => b[1].clarity - a[1].clarity)[0];
  const hardest = Object.entries(p.recovered).sort((a, b) => b[1].attempts - a[1].attempts)[0];
  return {
    recovered: entries.length,
    total: game.campaign.classes.length,
    skipped: p.skipped.filter(s => !p.recovered[s]),
    bestStreak: p.bestStreak,
    topConfusion: topConfusions(p, 1)[0] ?? null,
    clearest: clearest ? { name: clearest[0], clarity: clearest[1].clarity } : null,
    hardestWon: hardest ? { name: hardest[0], attempts: hardest[1].attempts } : null,
    attempts: p.attempts,
  };
}

export function Finale() {
  const game = useGame();
  const beats = useMemo(() => finaleBeats(finaleFacts(game)), []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { bus.emit('finale:phase', { phase: 'noise' }); }, []);
  return (
    <Cinematic beats={beats} speed={game.settings.textSpeed} integrity={1} ticks={game.campaign.classes.map(c => !!game.progress.recovered[c])}
      rings={game.campaign.zones.length} totalRings={game.campaign.zones.length} reducedMotion={game.reducedMotion}
      onBeat={b => { if (b.fx === 'converge' || b.fx === 'reboot' || b.fx === 'world') bus.emit('finale:phase', { phase: b.fx }); }}
      onDone={game.finishFinale} className="finale" />
  );
}

export function CampaignComplete() {
  const game = useGame();
  const { open } = useOverlay();
  const p = game.progress;
  const f = finaleFacts(game);
  const minutes = Math.max(1, Math.round((Date.now() - p.startedAt) / 60000));
  return (
    <main className="screen campaign-complete">
      <Wordmark torn={false} className="cc-mark" />
      <p className="cc-line">Memory recovery protocol complete.</p>
      <dl className="cc-stats">
        <div><dt>Memories restored</dt><dd>{f.recovered}<small> / {f.total}</small></dd></div>
        <div><dt>Score</dt><dd>{p.score.toLocaleString()}</dd></div>
        <div><dt>Best first-try streak</dt><dd>{p.bestStreak}</dd></div>
        <div><dt>Drawings sent to the network</dt><dd>{p.attempts}</dd></div>
        <div><dt>Times GLITCH said it was unsure</dt><dd>{p.uncertainCount}</dd></div>
        <div><dt>Rank</dt><dd className="cc-rank">{rankFor(p.xp).name}</dd></div>
      </dl>
      {topConfusions(p, 3).length > 0 && (
        <section className="cc-confusions">
          <h3>Where the network got confused</h3>
          <ul>{topConfusions(p, 3).map(c => <li key={c.target + c.predicted}>Your {c.target} read as {c.predicted} ×{c.count}</li>)}</ul>
        </section>
      )}
      <p className="cc-ach">{p.achievements.length} of {ACHIEVEMENTS.length} achievements · about {minutes} minutes</p>
      <div className="row">
        <Btn variant="primary" sound="confirm" autoFocus onClick={() => open('archive')}>Open the archive</Btn>
        <Btn onClick={game.toTitle}>Title screen</Btn>
      </div>
    </main>
  );
}
