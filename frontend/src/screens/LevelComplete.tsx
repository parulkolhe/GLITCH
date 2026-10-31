/** LEVEL_COMPLETE: the gate opens, a log fragment is recovered, the zone's drawings are shown together. */
import { useEffect, useState } from 'react';
import { useGame } from '../game/useGame';
import { Btn } from '../ui/Button';
import { StrokeArt } from '../ui/StrokeArt';
import { Glitch } from '../robot/Glitch';
import { ZONE_LOGS } from '../story/dialogue';
import { zoneBonus } from '../game/scoring';
import { useTypewriter } from '../ui/Speech';
import { fx } from '../fx/fxController';

export function LevelComplete() {
  const game = useGame();
  const zone = game.currentZone;
  const p = game.progress;
  const names = zone.final ? p.finalRecovered : zone.classes.filter(c => p.recovered[c]);
  const dark = zone.final ? [] : zone.classes.filter(c => !p.recovered[c]);
  const hints = p.hintsByZone[zone.id] ?? 0;
  const log = ZONE_LOGS[zone.id] ?? '';
  const { shown } = useTypewriter(log, game.settings.textSpeed, zone.id);
  const [ready, setReady] = useState(game.reducedMotion);
  useEffect(() => {
    fx.pulse(window.innerWidth / 2, window.innerHeight * 0.4, '#c8ff3d', 1.6);
    const id = setTimeout(() => setReady(true), 1200);
    return () => clearTimeout(id);
  }, []);
  const next = game.campaign.zones[game.campaign.zones.indexOf(zone) + 1];

  return (
    <main className="screen level-complete">
      <header>
        <p className="lc-kicker">Gate open</p>
        <h2>{zone.name}</h2>
        <p className="lc-sub">{names.length} memories restored here{dark.length ? `, ${dark.length} still dark` : ''}. Zone bonus +{zoneBonus(hints)}{hints === 0 ? ' (no echoes used)' : ''}.</p>
      </header>
      <ul className="lc-gallery">
        {names.map(n => <li key={n}><StrokeArt strokes={p.recovered[n]?.strokes ?? []} label={`Your ${n}`} /><span>{n}</span></li>)}
        {dark.map(n => <li key={n} className="is-dark"><span className="dark-glyph" aria-hidden="true">?</span><span>still dark</span></li>)}
      </ul>
      <aside className="lc-log"><span className="speech-name">Recovered log</span><p>{shown}</p></aside>
      <div className="lc-robot"><Glitch mood="excited" integrity={game.integrity} ticks={game.campaign.classes.map(c => !!p.recovered[c])}
        rings={p.zonesCompleted.length + 1} totalRings={game.campaign.zones.length} reducedMotion={game.reducedMotion} /></div>
      <footer>
        <Btn variant="primary" sound="confirm" disabled={!ready} autoFocus onClick={game.finishZone}>
          {zone.final ? 'Reboot the core' : next?.final ? 'Begin the core reboot' : `On to ${next?.name ?? 'the next zone'}`}
        </Btn>
      </footer>
    </main>
  );
}
