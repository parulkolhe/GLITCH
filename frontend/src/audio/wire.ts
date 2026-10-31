/** Connects gameplay events to sounds. The only place that decides "what sounds when". */
import { bus } from '../events/bus';
import { audio } from './engine';
import { AmbientMusic, ZONE_MUSIC } from './music';

export const music = new AmbientMusic(audio);
let drawingSounds = true;
export function setDrawingSounds(on: boolean) { drawingSounds = on; }

let wired = false;
export function wireAudio() {
  if (wired) return;
  wired = true;
  bus.on('ui:click', ({ variant }) => audio.play(variant === 'confirm' ? 'uiConfirm' : variant === 'back' ? 'uiBack' : 'uiClick'));
  bus.on('ui:hover', () => audio.play('uiHover'));
  bus.on('boot:start', () => audio.play('bootHum'));
  bus.on('boot:relay', () => audio.play('relay'));
  bus.on('boot:signature', () => { audio.play('bootRise'); setTimeout(() => audio.play('signature'), 650); });
  bus.on('cinematic:beat', ({ fx }) => {
    if (fx === 'signal') audio.play('scan');
    else if (fx === 'shatter') audio.play('shatter');
    else if (fx === 'log') audio.play('log');
    else if (fx === 'boot') audio.play('bootRise');
    else if (fx === 'core') audio.play('coreOnline');
  });
  bus.on('core:charge', ({ level }) => audio.play('coreCharge', { pitch: level }));
  bus.on('core:online', () => audio.play('coreOnline'));
  bus.on('zone:enter', ({ zoneId }) => {
    const m = ZONE_MUSIC[zoneId];
    if (m) { audio.keyRoot = m.root + 12; music.setParams(m); }
    audio.play('whoosh');
  });
  bus.on('draw:stroke-start', ({ y }) => { if (drawingSounds) audio.play('stroke', { pitch: 1 - y }); });
  bus.on('draw:clear', () => audio.play('clear'));
  bus.on('draw:undo', () => audio.play('undo'));
  bus.on('predict:start', () => { audio.play('scan'); audio.startTension(); });
  bus.on('predict:result', () => audio.stopTension());
  bus.on('predict:error', () => { audio.stopTension(); audio.play('error'); });
  bus.on('memory:recognized', ({ streak }) => {
    audio.play('recognized', { intensity: Math.min(streak, 6) });
    music.setParams({ intensity: Math.min(3, Math.floor(streak / 2) + 1) });
  });
  bus.on('memory:shard', () => audio.play('shard'));
  bus.on('memory:missed', ({ uncertain }) => {
    audio.play(uncertain ? 'uncertain' : 'miss');
    music.setParams({ intensity: 0 });
    setTimeout(() => audio.play('retry'), 520);
  });
  bus.on('memory:skipped', () => audio.play('uiBack'));
  bus.on('hint:reveal', () => audio.play('hint'));
  bus.on('streak:milestone', () => setTimeout(() => audio.play('streak'), 450));
  bus.on('achievement:unlock', () => setTimeout(() => audio.play('achievement'), 900));
  bus.on('zone:complete', () => audio.play('levelComplete'));
  bus.on('finale:phase', ({ phase }) => {
    if (phase === 'noise') { music.setParams({ corruption: 1, intensity: 0 }); audio.play('shatter'); }
    if (phase === 'converge') { music.setParams({ corruption: 0.5, intensity: 1 }); audio.play('whoosh'); }
    if (phase === 'reboot') { audio.play('coreOnline'); setTimeout(() => audio.play('levelComplete'), 900); }
    if (phase === 'world') { music.setScene('finale', { corruption: 0, intensity: 3, ...ZONE_MUSIC['core-reboot'] }); setTimeout(() => audio.play('signature'), 1200); }
  });
}
