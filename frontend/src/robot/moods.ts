/**
 * GLITCH's expressive states. Each mood is a full pose description consumed by
 * the SVG robot; transitions between moods are interpolated by CSS, and the
 * idle loop (blink, bob, antenna sway) runs on one requestAnimationFrame.
 */
export type Mood =
  | 'booting' | 'idle' | 'watching' | 'confused' | 'thinking' | 'scanning'
  | 'recognising' | 'excited' | 'frustrated' | 'recovering' | 'restored' | 'sleeping';

export interface Pose {
  /** Eye shape: open circle, happy arcs, squint lines, wide, question, scan bar, off. */
  eyes: 'open' | 'happy' | 'squint' | 'wide' | 'question' | 'scan' | 'off' | 'heart';
  eyeScale: number;
  /** Visor glyph shown under the eyes. */
  mouth: 'flat' | 'smile' | 'grin' | 'wobble' | 'o' | 'dots' | 'none';
  tilt: number;           // body tilt in degrees
  bob: number;            // idle bob amplitude in px
  antennaSpeed: number;   // sway speed multiplier
  glow: 'cyan' | 'lime' | 'coral' | 'violet' | 'ivory';
  shake: boolean;
}

export const POSES: Record<Mood, Pose> = {
  booting:     { eyes: 'off',      eyeScale: 1,    mouth: 'none',   tilt: -6, bob: 0.5, antennaSpeed: 0.2, glow: 'coral',  shake: false },
  sleeping:    { eyes: 'off',      eyeScale: 1,    mouth: 'none',   tilt: -10, bob: 0.3, antennaSpeed: 0.1, glow: 'coral', shake: false },
  idle:        { eyes: 'open',     eyeScale: 1,    mouth: 'flat',   tilt: 0,  bob: 4,   antennaSpeed: 1,   glow: 'cyan',   shake: false },
  watching:    { eyes: 'open',     eyeScale: 1.1,  mouth: 'o',      tilt: 3,  bob: 2,   antennaSpeed: 0.7, glow: 'cyan',   shake: false },
  confused:    { eyes: 'question', eyeScale: 1,    mouth: 'wobble', tilt: -9, bob: 2,   antennaSpeed: 1.8, glow: 'violet', shake: false },
  thinking:    { eyes: 'squint',   eyeScale: 0.9,  mouth: 'dots',   tilt: 6,  bob: 1.5, antennaSpeed: 2.2, glow: 'violet', shake: false },
  scanning:    { eyes: 'scan',     eyeScale: 1,    mouth: 'dots',   tilt: 0,  bob: 0.8, antennaSpeed: 3,   glow: 'cyan',   shake: false },
  recognising: { eyes: 'wide',     eyeScale: 1.25, mouth: 'o',      tilt: 0,  bob: 1,   antennaSpeed: 2,   glow: 'lime',   shake: false },
  excited:     { eyes: 'happy',    eyeScale: 1.1,  mouth: 'grin',   tilt: 4,  bob: 9,   antennaSpeed: 2.6, glow: 'lime',   shake: false },
  frustrated:  { eyes: 'squint',   eyeScale: 1,    mouth: 'wobble', tilt: -4, bob: 1.5, antennaSpeed: 0.8, glow: 'coral',  shake: true },
  recovering:  { eyes: 'heart',    eyeScale: 1.05, mouth: 'smile',  tilt: 0,  bob: 6,   antennaSpeed: 1.6, glow: 'lime',   shake: false },
  restored:    { eyes: 'happy',    eyeScale: 1,    mouth: 'smile',  tilt: 2,  bob: 5,   antennaSpeed: 1.1, glow: 'ivory',  shake: false },
};
