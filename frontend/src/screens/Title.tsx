import { useState } from 'react';
import { useGame } from '../game/useGame';
import { Glitch } from '../robot/Glitch';
import { Wordmark } from '../ui/Wordmark';
import { Btn } from '../ui/Button';
import { useOverlay } from '../ui/overlay';
import { rankFor } from '../game/scoring';

export function Title() {
  const game = useGame();
  const { open } = useOverlay();
  const [confirmNew, setConfirmNew] = useState(false);
  const p = game.progress;
  const recovered = Object.keys(p.recovered).length;
  const total = game.campaign.classes.length;
  const started = game.hasSave && (recovered > 0 || p.introSeen);
  const rushUnlocked = p.zonesCompleted.length >= 1;
  const ticks = game.campaign.classes.map(c => !!p.recovered[c]);
  const zone = game.currentZone;

  return (
    <main className="screen title">
      <div className="title-mark">
        <Wordmark />
        <p className="title-sub">A memory recovery protocol</p>
      </div>
      <div className="title-robot">
        <Glitch mood={started ? (game.integrity > 0.8 ? 'restored' : 'idle') : 'sleeping'} integrity={game.integrity} ticks={ticks}
          rings={p.zonesCompleted.length} totalRings={game.campaign.zones.length} reducedMotion={game.reducedMotion} />
      </div>
      <nav className="title-menu" aria-label="Main menu">
        {started && !p.campaignComplete && (
          <Btn variant="primary" sound="confirm" onClick={game.continueCampaign} autoFocus>
            Continue <span className="btn-detail">{zone.name}</span>
          </Btn>
        )}
        {!confirmNew ? (
          <Btn variant={started && !p.campaignComplete ? 'ghost' : 'primary'} sound="confirm" autoFocus={!started || p.campaignComplete}
            onClick={() => (started ? setConfirmNew(true) : game.startNewCampaign())}>
            {started ? 'Start a new recovery' : 'Begin recovery'}
          </Btn>
        ) : (
          <div className="confirm-row" role="group" aria-label="Confirm new recovery">
            <span>This erases your current campaign.</span>
            <Btn variant="primary" sound="confirm" onClick={game.startNewCampaign}>Erase and start</Btn>
            <Btn variant="quiet" sound="back" onClick={() => setConfirmNew(false)}>Keep it</Btn>
          </div>
        )}
        <Btn onClick={() => open('archive')}>Memory archive <span className="btn-detail">{recovered} of {total}</span></Btn>
        <Btn onClick={() => game.startPractice('rush')} disabled={!rushUnlocked || !game.model}
          title={rushUnlocked ? 'Recognise as many drawings as you can in 75 seconds' : 'Unlocks after the Boot Sector'}>
          Signal rush <span className="btn-detail">{rushUnlocked ? (p.rushBest ? `best ${p.rushBest}` : '75 seconds') : 'locked'}</span>
        </Btn>
        <Btn onClick={() => game.startPractice('practice')} disabled={!game.model}>Free drawing <span className="btn-detail">no score</span></Btn>
        <Btn variant="quiet" onClick={() => open('inside')}>Inside GLITCH’s head</Btn>
        <Btn variant="quiet" onClick={() => open('settings')}>Settings</Btn>
      </nav>
      {started && (
        <p className="title-status">
          {recovered} memories restored · {rankFor(p.xp).name} · {p.score.toLocaleString()} points
        </p>
      )}
      {!game.model && <p className="title-warning" role="status">Recognition core offline. You can look around, but drawings can’t be recognised until the backend is running.</p>}
    </main>
  );
}
