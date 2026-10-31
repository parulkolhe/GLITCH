/**
 * GLITCH's sound: every sound is synthesised at runtime with the Web Audio API.
 * There are no audio files.
 *
 * Signal flow
 *   voices -> sfx bus ─┐
 *   music  -> music bus ┼-> master -> compressor/limiter -> speakers
 *   (both buses also send to a procedurally generated reverb)
 *
 * Safety: the context is created only after a user gesture (autoplay rules);
 * a per-sound cooldown and a global voice cap stop overlapping effects from
 * turning into noise; the compressor keeps peaks under control.
 */

export type SoundName =
  | 'uiClick' | 'uiConfirm' | 'uiBack' | 'uiHover' | 'whoosh'
  | 'bootHum' | 'relay' | 'bootRise' | 'signature'
  | 'stroke' | 'clear' | 'undo'
  | 'scan' | 'recognized' | 'shard' | 'streak' | 'levelComplete' | 'achievement'
  | 'miss' | 'uncertain' | 'retry' | 'error' | 'hint' | 'coreCharge' | 'coreOnline'
  | 'typeTick' | 'log' | 'shatter';

const COOLDOWN_MS: Partial<Record<SoundName, number>> = {
  uiHover: 90, stroke: 70, typeTick: 45, shard: 35, relay: 60, uiClick: 40, coreCharge: 80,
};
const MAX_VOICES = 28;

export const NOTE = (midi: number) => 440 * Math.pow(2, (midi - 69) / 12);

export interface Volumes { master: number; music: number; sfx: number; muted: boolean }

export class AudioEngine {
  ctx: AudioContext | null = null;
  private master!: GainNode;
  music!: GainNode;
  private sfx!: GainNode;
  reverbSend!: GainNode;
  private noiseBuffer: AudioBuffer | null = null;
  private lastPlayed = new Map<SoundName, number>();
  private voices = 0;
  private tension: { osc: OscillatorNode; lfo: OscillatorNode; gain: GainNode } | null = null;
  private volumes: Volumes = { master: 0.8, music: 0.55, sfx: 0.8, muted: false };
  /** Semitone offset applied to musical stingers so they sit in the current zone's key. */
  keyRoot = 57; // A3

  get ready(): boolean { return !!this.ctx && this.ctx.state === 'running'; }

  /** Must be called from a user gesture (click / key / touch). Safe to call repeatedly. */
  unlock(): boolean {
    if (typeof window === 'undefined') return false;
    const Ctor: typeof AudioContext | undefined = (window as any).AudioContext || (window as any).webkitAudioContext;
    if (!Ctor) return false;
    if (!this.ctx) {
      try { this.ctx = new Ctor({ latencyHint: 'interactive' }); } catch { return false; }
      const ctx = this.ctx;
      const limiter = ctx.createDynamicsCompressor();
      limiter.threshold.value = -14; limiter.knee.value = 8; limiter.ratio.value = 6;
      limiter.attack.value = 0.004; limiter.release.value = 0.22;
      this.master = ctx.createGain();
      this.music = ctx.createGain();
      this.sfx = ctx.createGain();
      const reverb = ctx.createConvolver();
      reverb.buffer = this.impulse(2.6, 2.4);
      this.reverbSend = ctx.createGain();
      this.reverbSend.gain.value = 0.32;
      this.reverbSend.connect(reverb).connect(this.master);
      this.music.connect(this.master);
      this.sfx.connect(this.master);
      this.master.connect(limiter).connect(ctx.destination);
      this.applyVolumes();
      document.addEventListener('visibilitychange', () => {
        if (!this.ctx) return;
        if (document.hidden) this.ctx.suspend().catch(() => {}); else this.ctx.resume().catch(() => {});
      });
    }
    if (this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
    return true;
  }

  setVolumes(v: Volumes) { this.volumes = v; this.applyVolumes(); }

  private applyVolumes() {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    const m = this.volumes.muted ? 0 : this.volumes.master;
    // perceptual curve: squares feel more even on a slider than linear gain
    this.master.gain.setTargetAtTime(m * m * 0.9, t, 0.05);
    this.music.gain.setTargetAtTime(this.volumes.music * this.volumes.music, t, 0.05);
    this.sfx.gain.setTargetAtTime(this.volumes.sfx * this.volumes.sfx, t, 0.05);
  }

  // ------------------------------------------------------------ primitives
  private impulse(seconds: number, decay: number): AudioBuffer {
    const ctx = this.ctx!;
    const length = Math.floor(ctx.sampleRate * seconds);
    const buffer = ctx.createBuffer(2, length, ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const data = buffer.getChannelData(ch);
      for (let i = 0; i < length; i++) data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, decay);
    }
    return buffer;
  }

  noise(): AudioBuffer {
    if (!this.noiseBuffer) {
      const ctx = this.ctx!;
      this.noiseBuffer = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
      const d = this.noiseBuffer.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    }
    return this.noiseBuffer;
  }

  private gate(name: SoundName): boolean {
    if (!this.ready) return false;
    const now = performance.now();
    const cd = COOLDOWN_MS[name] ?? 25;
    if (now - (this.lastPlayed.get(name) ?? -1e9) < cd) return false;
    if (this.voices >= MAX_VOICES && name !== 'recognized' && name !== 'levelComplete') return false;
    this.lastPlayed.set(name, now);
    return true;
  }

  private track(node: AudioScheduledSourceNode) {
    this.voices++;
    node.onended = () => { this.voices = Math.max(0, this.voices - 1); };
  }

  /** One enveloped oscillator note. */
  tone(freq: number, at: number, dur: number, o: {
    type?: OscillatorType; gain?: number; attack?: number; release?: number; detune?: number;
    to?: number; filter?: number; q?: number; dest?: AudioNode; send?: number; pan?: number;
  } = {}) {
    const ctx = this.ctx!;
    const osc = ctx.createOscillator();
    osc.type = o.type ?? 'sine';
    osc.frequency.setValueAtTime(freq, at);
    if (o.to) osc.frequency.exponentialRampToValueAtTime(Math.max(20, o.to), at + dur);
    if (o.detune) osc.detune.value = o.detune;
    const env = ctx.createGain();
    const peak = o.gain ?? 0.2;
    const attack = o.attack ?? 0.005;
    const release = o.release ?? Math.max(0.03, dur * 0.6);
    env.gain.setValueAtTime(0.0001, at);
    env.gain.exponentialRampToValueAtTime(peak, at + attack);
    env.gain.setValueAtTime(peak, at + Math.max(attack, dur - release));
    env.gain.exponentialRampToValueAtTime(0.0001, at + dur);
    let node: AudioNode = osc;
    if (o.filter) {
      const f = ctx.createBiquadFilter();
      f.type = 'lowpass'; f.frequency.value = o.filter; f.Q.value = o.q ?? 0.7;
      node.connect(f); node = f;
    }
    node.connect(env);
    let out: AudioNode = env;
    if (o.pan) { const p = ctx.createStereoPanner(); p.pan.value = o.pan; env.connect(p); out = p; }
    out.connect(o.dest ?? this.sfx);
    if (o.send) { const s = ctx.createGain(); s.gain.value = o.send; out.connect(s).connect(this.reverbSend); }
    osc.start(at);
    osc.stop(at + dur + 0.05);
    if (!o.dest) this.track(osc); // only sound effects count toward the voice cap
    return osc;
  }

  /** Filtered noise burst. */
  burst(at: number, dur: number, o: { gain?: number; type?: BiquadFilterType; freq?: number; to?: number; q?: number; dest?: AudioNode; send?: number; attack?: number } = {}) {
    const ctx = this.ctx!;
    const src = ctx.createBufferSource();
    src.buffer = this.noise();
    src.loop = true;
    const f = ctx.createBiquadFilter();
    f.type = o.type ?? 'bandpass';
    f.frequency.setValueAtTime(o.freq ?? 2000, at);
    if (o.to) f.frequency.exponentialRampToValueAtTime(o.to, at + dur);
    f.Q.value = o.q ?? 1;
    const env = ctx.createGain();
    env.gain.setValueAtTime(0.0001, at);
    env.gain.exponentialRampToValueAtTime(o.gain ?? 0.1, at + (o.attack ?? 0.004));
    env.gain.exponentialRampToValueAtTime(0.0001, at + dur);
    src.connect(f).connect(env).connect(o.dest ?? this.sfx);
    if (o.send) { const s = ctx.createGain(); s.gain.value = o.send; env.connect(s).connect(this.reverbSend); }
    src.start(at, Math.random() * 1.5);
    src.stop(at + dur + 0.05);
    if (!o.dest) this.track(src);
  }

  /** Two-operator FM bell: the "recognition" timbre. */
  bell(freq: number, at: number, dur: number, gain = 0.12, ratio = 3.5, index = 2.2, send = 0.5) {
    const ctx = this.ctx!;
    const carrier = ctx.createOscillator();
    const mod = ctx.createOscillator();
    const modGain = ctx.createGain();
    carrier.frequency.value = freq;
    mod.frequency.value = freq * ratio;
    modGain.gain.setValueAtTime(freq * index, at);
    modGain.gain.exponentialRampToValueAtTime(freq * 0.05, at + dur);
    mod.connect(modGain).connect(carrier.frequency);
    const env = ctx.createGain();
    env.gain.setValueAtTime(0.0001, at);
    env.gain.exponentialRampToValueAtTime(gain, at + 0.006);
    env.gain.exponentialRampToValueAtTime(0.0001, at + dur);
    carrier.connect(env).connect(this.sfx);
    const s = ctx.createGain(); s.gain.value = send; env.connect(s).connect(this.reverbSend);
    carrier.start(at); mod.start(at);
    carrier.stop(at + dur + 0.05); mod.stop(at + dur + 0.05);
    this.track(carrier);
  }

  // --------------------------------------------------------------- library
  play(name: SoundName, opts: { intensity?: number; pitch?: number } = {}) {
    if (!this.gate(name)) return;
    const ctx = this.ctx!;
    const t = ctx.currentTime + 0.005;
    const k = this.keyRoot;
    const intensity = opts.intensity ?? 0;
    switch (name) {
      case 'uiHover': this.tone(NOTE(k + 31), t, 0.04, { type: 'sine', gain: 0.018 }); break;
      case 'uiClick':
        this.tone(NOTE(k + 24), t, 0.05, { type: 'square', gain: 0.03, filter: 2400 });
        this.burst(t, 0.02, { freq: 5000, gain: 0.03, q: 3 });
        break;
      case 'uiConfirm':
        this.tone(NOTE(k + 19), t, 0.08, { type: 'triangle', gain: 0.08 });
        this.tone(NOTE(k + 26), t + 0.06, 0.14, { type: 'triangle', gain: 0.08, send: 0.3 });
        break;
      case 'uiBack': this.tone(NOTE(k + 21), t, 0.12, { type: 'triangle', gain: 0.06, to: NOTE(k + 14) }); break;
      case 'whoosh': this.burst(t, 0.7, { type: 'bandpass', freq: 300, to: 3500, q: 2.5, gain: 0.07, attack: 0.25, send: 0.6 }); break;
      case 'bootHum': {
        this.tone(55, t, 3.2, { type: 'sine', gain: 0.22, attack: 1.2, release: 1.2 });
        this.tone(110, t, 3.2, { type: 'sawtooth', gain: 0.05, attack: 1.4, release: 1, filter: 260, detune: -8 });
        this.tone(110.6, t, 3.2, { type: 'sawtooth', gain: 0.05, attack: 1.4, release: 1, filter: 260, detune: 9 });
        break;
      }
      case 'relay':
        this.burst(t, 0.03, { freq: 1800 + Math.random() * 1500, q: 8, gain: 0.12 });
        this.tone(70, t, 0.06, { type: 'sine', gain: 0.12, to: 40 });
        break;
      case 'bootRise': {
        const steps = [0, 3, 7, 10, 12, 15, 19, 24];
        steps.forEach((s, i) => this.tone(NOTE(k + s), t + i * 0.07, 0.09, { type: 'square', gain: 0.035, filter: 1200 + i * 400 }));
        this.tone(NOTE(k - 12), t, 0.7, { type: 'sine', gain: 0.1, to: NOTE(k), attack: 0.3 });
        break;
      }
      case 'signature': {
        // GLITCH's motif: a rising fourth, a falling third, then a stuttered high note.
        const motif = [[0, 0.16], [5, 0.16], [12, 0.16], [9, 0.34]] as const;
        let at = t;
        for (const [s, d] of motif) { this.bell(NOTE(k + 12 + s), at, d * 3, 0.07, 2, 1.4, 0.6); this.tone(NOTE(k + s), at, d, { type: 'square', gain: 0.03, filter: 1600 }); at += d; }
        for (let i = 0; i < 3; i++) this.tone(NOTE(k + 28), at + i * 0.045, 0.035, { type: 'square', gain: 0.03, filter: 3000 });
        this.burst(at, 0.12, { freq: 6000, q: 0.8, gain: 0.03 });
        break;
      }
      case 'stroke': {
        const p = opts.pitch ?? 0.5;
        this.burst(t, 0.05, { freq: 1400 + p * 2200, q: 6, gain: 0.012 });
        break;
      }
      case 'clear': this.burst(t, 0.35, { type: 'highpass', freq: 600, to: 5000, gain: 0.05, attack: 0.05 }); break;
      case 'undo': this.tone(NOTE(k + 17), t, 0.08, { type: 'triangle', gain: 0.05, to: NOTE(k + 12) }); break;
      case 'scan': {
        this.tone(220, t, 0.55, { type: 'sine', gain: 0.07, to: 1320, filter: 2400, send: 0.3 });
        for (let i = 0; i < 6; i++) this.tone(NOTE(k + 24 + (i % 2) * 7), t + i * 0.08, 0.03, { type: 'square', gain: 0.02, filter: 2500 });
        break;
      }
      case 'recognized': {
        const lift = Math.min(intensity, 6);
        const chord = [0, 4, 7, 12, 16].map(s => s + lift);
        chord.forEach((s, i) => this.bell(NOTE(k + 12 + s), t + i * 0.055, 1.6, 0.075));
        this.tone(NOTE(k - 12), t, 0.6, { type: 'sine', gain: 0.18, to: NOTE(k - 24), attack: 0.005 });
        this.burst(t + 0.1, 0.9, { type: 'highpass', freq: 7000, gain: 0.025, attack: 0.2, send: 0.7 });
        break;
      }
      case 'shard': {
        const pent = [0, 2, 4, 7, 9];
        const s = pent[Math.floor(Math.random() * pent.length)] + 24 + 12 * Math.floor(Math.random() * 2);
        this.tone(NOTE(k + s), t, 0.25, { type: 'sine', gain: 0.035, send: 0.6, pan: Math.random() * 1.4 - 0.7 });
        break;
      }
      case 'streak': {
        [0, 7, 12, 16, 19, 24].forEach((s, i) => this.tone(NOTE(k + 12 + s), t + i * 0.045, 0.5, { type: 'triangle', gain: 0.05, send: 0.5 }));
        this.burst(t, 0.5, { type: 'bandpass', freq: 800, to: 8000, q: 1.2, gain: 0.04, attack: 0.2 });
        break;
      }
      case 'levelComplete': {
        const prog = [[0, 4, 7], [5, 9, 12], [7, 11, 14], [12, 16, 19, 24]];
        prog.forEach((ch, i) => ch.forEach(s => this.tone(NOTE(k + s), t + i * 0.38, i === 3 ? 2.4 : 0.5,
          { type: 'sawtooth', gain: 0.025, filter: 1800, attack: 0.04, send: 0.5, detune: (Math.random() - 0.5) * 12 })));
        prog.forEach((ch, i) => this.bell(NOTE(k + 12 + ch[ch.length - 1]), t + i * 0.38, 1.2, 0.05));
        this.tone(NOTE(k - 12), t + 1.14, 2.2, { type: 'sine', gain: 0.14, attack: 0.05 });
        break;
      }
      case 'achievement':
        [12, 19, 24].forEach((s, i) => this.bell(NOTE(k + 12 + s), t + i * 0.09, 1, 0.05, 1.5, 1));
        break;
      case 'miss': {
        // a soft, slightly comic "hm-mm": never a buzzer
        this.tone(NOTE(k + 7), t, 0.16, { type: 'triangle', gain: 0.07, to: NOTE(k + 5), filter: 1600 });
        this.tone(NOTE(k + 3), t + 0.15, 0.24, { type: 'triangle', gain: 0.06, to: NOTE(k + 1), filter: 1400 });
        for (let i = 0; i < 4; i++) this.burst(t + 0.05 + i * 0.03, 0.02, { freq: 3000 + Math.random() * 3000, q: 10, gain: 0.02 });
        break;
      }
      case 'uncertain':
        this.tone(NOTE(k + 5), t, 0.14, { type: 'sine', gain: 0.07 });
        this.tone(NOTE(k + 9), t + 0.13, 0.26, { type: 'sine', gain: 0.07, to: NOTE(k + 11), send: 0.3 });
        break;
      case 'retry': this.tone(NOTE(k + 12), t, 0.1, { type: 'sine', gain: 0.05, send: 0.2 }); break;
      case 'error':
        this.tone(NOTE(k), t, 0.3, { type: 'square', gain: 0.03, filter: 700, to: NOTE(k - 5) });
        this.burst(t, 0.25, { freq: 900, q: 0.8, gain: 0.04 });
        break;
      case 'hint':
        [0, 7, 14].forEach((s, i) => this.tone(NOTE(k + 19 + s), t + i * 0.12, 0.9, { type: 'sine', gain: 0.03, send: 0.8 }));
        break;
      case 'coreCharge': {
        const p = Math.min(1, Math.max(0, opts.pitch ?? 0));
        this.tone(NOTE(k + 12 + p * 24), t, 0.09, { type: 'triangle', gain: 0.03 + 0.03 * p, filter: 3000 });
        break;
      }
      case 'coreOnline':
        this.tone(NOTE(k - 24), t, 2.5, { type: 'sine', gain: 0.2, attack: 0.02 });
        [0, 7, 12, 19, 24].forEach((s, i) => this.bell(NOTE(k + s), t + i * 0.08, 2.5, 0.06, 2, 1.2));
        this.burst(t, 1.4, { type: 'lowpass', freq: 200, to: 9000, gain: 0.06, attack: 0.02, send: 0.8 });
        break;
      case 'typeTick': this.burst(t, 0.012, { freq: 4200 + Math.random() * 1200, q: 4, gain: 0.012 }); break;
      case 'log':
        for (let i = 0; i < 8; i++) this.tone(NOTE(k + 24 + ((i * 5) % 12)), t + i * 0.03, 0.025, { type: 'square', gain: 0.018, filter: 3000 });
        break;
      case 'shatter':
        this.burst(t, 1.6, { type: 'lowpass', freq: 8000, to: 120, q: 0.5, gain: 0.2, attack: 0.003, send: 0.6 });
        for (let i = 0; i < 14; i++) this.tone(200 + Math.random() * 3000, t + Math.random() * 0.5, 0.05, { type: 'square', gain: 0.03, filter: 4000 });
        this.tone(80, t, 1.2, { type: 'sawtooth', gain: 0.08, to: 30, filter: 400 });
        break;
    }
  }

  /** Rising tension under the scan; stops the moment the result is known. */
  startTension() {
    if (!this.ready || this.tension) return;
    const ctx = this.ctx!;
    const t = ctx.currentTime;
    const osc = ctx.createOscillator();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(NOTE(this.keyRoot - 12), t);
    osc.frequency.linearRampToValueAtTime(NOTE(this.keyRoot), t + 3);
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass'; filter.frequency.setValueAtTime(300, t); filter.frequency.linearRampToValueAtTime(1800, t + 3); filter.Q.value = 6;
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(0.035, t + 0.4);
    const lfo = ctx.createOscillator();
    const lfoGain = ctx.createGain();
    lfo.frequency.value = 7; lfoGain.gain.value = 0.015;
    lfo.connect(lfoGain).connect(gain.gain);
    osc.connect(filter).connect(gain).connect(this.sfx);
    osc.start(t); lfo.start(t);
    this.tension = { osc, lfo, gain };
  }

  stopTension() {
    if (!this.tension || !this.ctx) return;
    const { osc, lfo, gain } = this.tension;
    const t = this.ctx.currentTime;
    gain.gain.cancelScheduledValues(t);
    gain.gain.setTargetAtTime(0.0001, t, 0.03);
    osc.stop(t + 0.2); lfo.stop(t + 0.2);
    this.tension = null;
  }
}

export const audio = new AudioEngine();
