/**
 * Procedural ambient score. A lookahead scheduler (Chris Wilson's pattern)
 * plays a slow chord progression on a detuned pad, a sub pulse, a plucked
 * arpeggio and soft ticks. Game state shapes it:
 *
 *   corruption (0..1)  random detune, bit-noise and dropouts; the pad filter closes
 *   intensity  (0..3)  layers enter with the recognition streak
 *   mode               each zone has its own scale and root
 *
 * Scenes crossfade over a couple of seconds rather than cutting.
 */
import { AudioEngine, NOTE } from './engine';

export type Scene = 'silence' | 'title' | 'cinematic' | 'zone' | 'finale';

export interface MusicParams {
  root: number;        // MIDI note
  scale: number[];     // semitone offsets
  corruption: number;  // 0..1
  intensity: number;   // 0..3
  tempo: number;       // bpm
}

export const SCALES = {
  aeolian: [0, 2, 3, 5, 7, 8, 10],
  dorian: [0, 2, 3, 5, 7, 9, 10],
  lydian: [0, 2, 4, 6, 7, 9, 11],
  mixolydian: [0, 2, 4, 5, 7, 9, 10],
  phrygian: [0, 1, 3, 5, 7, 8, 10],
  ionian: [0, 2, 4, 5, 7, 9, 11],
};

/** Scale degrees (0-based) of each chord's root; four bars per cycle. */
const PROGRESSION = [0, 5, 3, 4];

export class AmbientMusic {
  private timer: ReturnType<typeof setInterval> | null = null;
  private step = 0;
  private nextTime = 0;
  private bus: GainNode | null = null;
  private params: MusicParams = { root: 45, scale: SCALES.aeolian, corruption: 1, intensity: 0, tempo: 66 };
  private scene: Scene = 'silence';

  constructor(private engine: AudioEngine) {}

  setParams(p: Partial<MusicParams>) { this.params = { ...this.params, ...p }; }
  get current(): Scene { return this.scene; }

  setScene(scene: Scene, p: Partial<MusicParams> = {}) {
    this.setParams(p);
    if (scene === this.scene) return;
    this.scene = scene;
    const ctx = this.engine.ctx;
    if (!ctx) return;
    if (scene === 'silence') { this.fadeOut(); return; }
    if (!this.bus) {
      this.bus = ctx.createGain();
      this.bus.gain.value = 0.0001;
      this.bus.connect(this.engine.music);
    }
    const level = scene === 'cinematic' ? 0.55 : scene === 'title' ? 0.8 : 1;
    this.bus.gain.cancelScheduledValues(ctx.currentTime);
    this.bus.gain.setTargetAtTime(level, ctx.currentTime, 0.9);
    if (!this.timer) {
      this.nextTime = ctx.currentTime + 0.1;
      this.timer = setInterval(() => this.schedule(), 90);
    }
  }

  private fadeOut() {
    const ctx = this.engine.ctx;
    if (!ctx || !this.bus) return;
    this.bus.gain.setTargetAtTime(0.0001, ctx.currentTime, 0.6);
    setTimeout(() => { if (this.scene === 'silence' && this.timer) { clearInterval(this.timer); this.timer = null; } }, 3000);
  }

  stop() { if (this.timer) clearInterval(this.timer); this.timer = null; }

  private degree(d: number, octave = 0): number {
    const s = this.params.scale;
    const idx = ((d % s.length) + s.length) % s.length;
    return this.params.root + s[idx] + 12 * (octave + Math.floor(d / s.length));
  }

  private schedule() {
    const ctx = this.engine.ctx;
    if (!ctx || !this.bus || ctx.state !== 'running') return;
    const sixteenth = 60 / this.params.tempo / 4;
    while (this.nextTime < ctx.currentTime + 0.35) {
      this.playStep(this.step, this.nextTime, sixteenth);
      this.nextTime += sixteenth;
      this.step = (this.step + 1) % (16 * PROGRESSION.length);
    }
  }

  private playStep(step: number, t: number, sixteenth: number) {
    const e = this.engine;
    const bus = this.bus!;
    const { corruption, intensity } = this.params;
    const bar = Math.floor(step / 16);
    const inBar = step % 16;
    const chordRoot = PROGRESSION[bar];
    const finale = this.scene === 'finale';
    const drop = () => Math.random() < corruption * 0.35;

    // Pad: a three-voice chord at the top of each bar, long and detuned.
    if (inBar === 0) {
      const barLen = sixteenth * 16;
      const cutoff = 380 + (1 - corruption) * 1800 + (finale ? 1200 : 0);
      for (const d of [0, 2, 4, finale ? 6 : 7]) {
        const freq = NOTE(this.degree(chordRoot + d, 0));
        for (const detune of [-7, 7]) {
          e.tone(freq, t, barLen * 1.15, { type: 'sawtooth', gain: 0.016, attack: barLen * 0.35, release: barLen * 0.5,
            filter: cutoff, detune: detune + (Math.random() - 0.5) * 30 * corruption, dest: bus, send: 0.55 });
        }
      }
      e.tone(NOTE(this.degree(chordRoot, -1)), t, barLen, { type: 'sine', gain: 0.07, attack: 0.4, release: barLen * 0.4, dest: bus });
    }

    // Sub pulse on beats one and three once a streak begins.
    if (intensity >= 2 && (inBar === 0 || inBar === 8) && !drop()) {
      e.tone(NOTE(this.degree(chordRoot, -1)), t, 0.35, { type: 'sine', gain: 0.12, to: NOTE(this.degree(chordRoot, -2)), dest: bus });
    }

    // Arpeggio: eighth notes walking up the chord, enters with intensity 1.
    if (intensity >= 1 && inBar % 2 === 0 && !drop()) {
      const pattern = [0, 2, 4, 7, 4, 2, 4, 9];
      const d = chordRoot + pattern[(inBar / 2) % pattern.length];
      e.tone(NOTE(this.degree(d, 1)), t, sixteenth * 1.8, { type: 'triangle', gain: 0.022 + 0.008 * intensity, filter: 2600, dest: bus, send: 0.45,
        pan: Math.sin(step * 0.7) * 0.5 });
    }

    // Ticks: soft off-beat noise, the top layer.
    if (intensity >= 3 && inBar % 4 === 2) {
      e.burst(t, 0.03, { type: 'highpass', freq: 8000, gain: 0.018, dest: bus });
    }

    // A sparse melody quoting GLITCH's motif when the world is mostly healed.
    if (corruption < 0.45 && inBar === 6 && bar % 2 === 1 && Math.random() < 0.6) {
      const motif = [0, 3, 7, 5];
      motif.forEach((m, i) => e.tone(NOTE(this.degree(chordRoot + m, 1)), t + i * sixteenth * 2, sixteenth * 2.5,
        { type: 'sine', gain: 0.028, dest: bus, send: 0.7 }));
    }

    // Corruption: bit noise and wrong notes. Fades out as memories return.
    if (Math.random() < corruption * 0.09) {
      e.burst(t, 0.04 + Math.random() * 0.12, { freq: 500 + Math.random() * 6000, q: 12, gain: 0.02 * corruption, dest: bus });
    }
    if (Math.random() < corruption * 0.03) {
      e.tone(200 + Math.random() * 1600, t, 0.06, { type: 'square', gain: 0.012, filter: 3000, dest: bus });
    }
  }
}

/** Musical identity per zone. */
export const ZONE_MUSIC: Record<string, { root: number; scale: number[]; tempo: number }> = {
  'boot-sector': { root: 45, scale: SCALES.dorian, tempo: 64 },
  'lost-basics': { root: 48, scale: SCALES.mixolydian, tempo: 70 },
  'forgotten-world': { root: 43, scale: SCALES.lydian, tempo: 62 },
  'signal-break': { root: 47, scale: SCALES.aeolian, tempo: 76 },
  'living-archive': { root: 44, scale: SCALES.phrygian, tempo: 68 },
  'core-reboot': { root: 45, scale: SCALES.ionian, tempo: 80 },
};
