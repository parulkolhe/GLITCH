import { useGame } from '../game/useGame';
import { Cinematic } from './Cinematic';
import { ACTS } from '../game/campaign';
import { ZONE_INTROS } from '../story/dialogue';
import { useEffect } from 'react';
import { music } from '../audio/wire';
import { ZONE_MUSIC } from '../audio/music';

export function ZoneIntro() {
  const game = useGame();
  const zone = game.currentZone;
  const index = game.campaign.zones.indexOf(zone);
  const firstOfAct = index === 0 || game.campaign.zones[index - 1]?.act !== zone.act;
  const beats = ZONE_INTROS[zone.id] ?? [{ speaker: 'glitch' as const, mood: 'idle' as const, text: `${zone.name}. ${zone.subtitle}` }];
  const p = game.progress;
  useEffect(() => { music.setScene('cinematic', { corruption: 1 - game.integrity, ...(ZONE_MUSIC[zone.id] ?? {}) }); }, [zone.id]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <Cinematic key={zone.id} beats={beats} speed={game.settings.textSpeed} integrity={game.integrity}
      ticks={game.campaign.classes.map(c => !!p.recovered[c])} rings={p.zonesCompleted.length} totalRings={game.campaign.zones.length}
      reducedMotion={game.reducedMotion} onDone={game.enterZone}
      kicker={firstOfAct ? ACTS[zone.act].title : `Memory zone ${index + 1} of ${game.campaign.zones.length}`}
      title={zone.name} className={`zone-intro zone-${zone.id}`} />
  );
}
