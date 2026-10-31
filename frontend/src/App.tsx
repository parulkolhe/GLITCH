import { useEffect, useMemo, useState } from 'react';
import { GameProvider, useGame } from './game/useGame';
import { World, type WorldProps } from './fx/World';
import { OverlayContext, type OverlayName } from './ui/overlay';
import { Boot } from './screens/Boot';
import { Title } from './screens/Title';
import { Cinematic } from './screens/Cinematic';
import { CoreActivation } from './screens/CoreActivation';
import { ZoneIntro } from './screens/ZoneIntro';
import { Play } from './screens/Play';
import { LevelComplete } from './screens/LevelComplete';
import { Finale, CampaignComplete } from './screens/Finale';
import { Archive, Inside, Menu, Settings } from './screens/Overlays';
import { INTRO } from './story/dialogue';
import { bus } from './events/bus';
import { audio } from './audio/engine';
import { music, setDrawingSounds, wireAudio } from './audio/wire';
import { Btn } from './ui/Button';
import { IconMenu, IconMute, IconSound } from './ui/Icons';
import { achievementById } from './game/achievements';
import { isDrawingPhase } from './game/stateMachine';

wireAudio();

function Hud({ onMenu }: { onMenu(): void }) {
  const game = useGame();
  const p = game.progress;
  const total = game.campaign.classes.length;
  const recovered = Object.keys(p.recovered).length;
  return (
    <header className="hud">
      <div className="hud-left">
        <span className="hud-mark" aria-hidden="true">GLITCH</span>
        {game.mode === 'campaign' && <span className="hud-zone">{game.currentZone.name}</span>}
      </div>
      <div className="hud-right">
        {game.mode === 'campaign' && (
          <>
            <span className="hud-memories" title={`${recovered} of ${total} memories restored`}>
              <span className="hud-meter" style={{ ['--fill' as any]: recovered / Math.max(1, total) }} aria-hidden="true" />
              {recovered}/{total}
            </span>
            <span className="hud-score">{p.score.toLocaleString()} pts</span>
            {p.streak >= 2 && <span className="hud-streak">streak {p.streak}</span>}
          </>
        )}
        <Btn variant="icon" sound="none" aria-label={game.settings.muted ? 'Unmute' : 'Mute'} aria-pressed={game.settings.muted}
          onClick={() => game.updateSettings({ muted: !game.settings.muted })}>
          {game.settings.muted ? <IconMute /> : <IconSound />}
        </Btn>
        <Btn variant="icon" aria-label="Pause menu" onClick={onMenu}><IconMenu /></Btn>
      </div>
    </header>
  );
}

function Toast() {
  const { toast } = useGame();
  const [visible, setVisible] = useState<typeof toast>(null);
  useEffect(() => {
    if (!toast) return;
    setVisible(toast);
    const id = setTimeout(() => setVisible(null), 3600);
    return () => clearTimeout(id);
  }, [toast]);
  const a = visible ? achievementById(visible.id) : null;
  return (
    <div className="toast-region" aria-live="polite">
      {a && <div key={visible!.key} className="toast"><strong>{a.name}</strong><span>{a.description}</span></div>}
    </div>
  );
}

function Shell() {
  const game = useGame();
  const [overlay, setOverlay] = useState<OverlayName>(null);
  const [beat, setBeat] = useState<string | undefined>(undefined);
  const { phase } = game;

  useEffect(() => bus.on('cinematic:beat', ({ fx }) => { if (fx) setBeat(fx); }), []);
  useEffect(() => { if (phase !== 'INTRO_CINEMATIC' && phase !== 'FINAL_SEQUENCE') setBeat(undefined); }, [phase]);

  // keep the audio engine in step with settings
  useEffect(() => {
    const s = game.settings;
    audio.setVolumes({ master: s.masterVolume, music: s.musicVolume, sfx: s.sfxVolume, muted: s.muted });
    setDrawingSounds(s.drawingSounds);
  }, [game.settings]);

  useEffect(() => {
    if (phase === 'TITLE') music.setScene('title', { corruption: 1 - game.integrity, intensity: 0 });
    if (phase === 'INTRO_CINEMATIC') music.setScene('cinematic', { corruption: 0.2 });
    if (phase === 'CAMPAIGN_COMPLETE') music.setScene('finale', { corruption: 0, intensity: 2 });
  }, [phase]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    document.documentElement.dataset.motion = game.reducedMotion ? 'reduced' : 'full';
  }, [game.reducedMotion]);

  // Esc opens the pause menu during play (cinematics use Esc to skip)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || overlay) return;
      if (isDrawingPhase(phase) || phase === 'LEVEL_COMPLETE' || phase === 'CAMPAIGN_COMPLETE') { e.preventDefault(); setOverlay('menu'); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [phase, overlay]);

  const world: WorldProps = useMemo(() => {
    const p = game.progress;
    const zone = game.currentZone;
    const constellations = Object.entries(p.recovered).map(([name, m]) => ({ name, strokes: m.strokes, addedAt: Date.now() - m.at < 6000 ? m.at : undefined }));
    const corruption = 1 - game.integrity;
    if (phase === 'BOOT') return { scene: 'cinematic', beat: 'silence', corruption: 1, hue: 200, grid: 'flat', constellations: [], reducedMotion: game.reducedMotion };
    if (phase === 'INTRO_CINEMATIC' || phase === 'MEMORY_CORE_ACTIVATION')
      return { scene: 'cinematic', beat: phase === 'MEMORY_CORE_ACTIVATION' ? 'flicker' : beat, corruption: 1, hue: 195, grid: 'flat', constellations: [], reducedMotion: game.reducedMotion };
    if (phase === 'FINAL_SEQUENCE' || phase === 'CAMPAIGN_COMPLETE')
      return { scene: 'finale', beat: phase === 'CAMPAIGN_COMPLETE' ? 'world' : beat, corruption: phase === 'CAMPAIGN_COMPLETE' ? 0 : corruption, hue: 78, grid: 'core', constellations, reducedMotion: game.reducedMotion };
    if (phase === 'TITLE') return { scene: 'title', corruption, hue: 195, grid: 'flat', constellations, reducedMotion: game.reducedMotion };
    const practice = game.mode === 'practice' || game.mode === 'rush';
    return { scene: 'zone', corruption: practice ? 0.15 : corruption, hue: practice ? 190 : zone.hue, grid: practice ? 'flat' : zone.grid, constellations, reducedMotion: game.reducedMotion };
  }, [phase, beat, game.progress, game.currentZone, game.integrity, game.mode, game.reducedMotion]);

  const showHud = !['BOOT', 'TITLE', 'INTRO_CINEMATIC', 'FINAL_SEQUENCE'].includes(phase);

  let screen: React.ReactNode = null;
  switch (phase) {
    case 'BOOT': screen = <Boot />; break;
    case 'TITLE': screen = <Title />; break;
    case 'INTRO_CINEMATIC':
      screen = <Cinematic beats={INTRO} speed={game.settings.textSpeed} integrity={0} ticks={[]} rings={0} totalRings={game.campaign.zones.length}
        reducedMotion={game.reducedMotion} onDone={() => game.send('INTRO_DONE')} showRobotFrom={7} className="intro" />;
      break;
    case 'MEMORY_CORE_ACTIVATION': screen = <CoreActivation />; break;
    case 'ZONE_INTRO': screen = <ZoneIntro />; break;
    case 'LEVEL_COMPLETE': screen = <LevelComplete />; break;
    case 'FINAL_SEQUENCE': screen = <Finale />; break;
    case 'CAMPAIGN_COMPLETE': screen = <CampaignComplete />; break;
    default: screen = <Play />;
  }

  return (
    <OverlayContext.Provider value={{ overlay, open: setOverlay }}>
      <World {...world} />
      <div className="scanlines" aria-hidden="true" />
      <div className={`app phase-${phase.toLowerCase()}`}>
        {showHud && <Hud onMenu={() => setOverlay('menu')} />}
        {screen}
      </div>
      {overlay === 'archive' && <Archive />}
      {overlay === 'settings' && <Settings />}
      {overlay === 'inside' && <Inside />}
      {overlay === 'menu' && <Menu />}
      <Toast />
    </OverlayContext.Provider>
  );
}

export function App() {
  return <GameProvider><Shell /></GameProvider>;
}
