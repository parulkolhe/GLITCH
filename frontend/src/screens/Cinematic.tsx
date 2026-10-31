/**
 * Plays a list of beats: narration, GLITCH dialogue, logs and choices. Each
 * beat can trigger a world FX scene and a robot mood. Space / Enter / click
 * advances; Esc skips the whole scene.
 */
import { useCallback, useEffect, useState } from 'react';
import type { Beat } from '../story/dialogue';
import { useTypewriter } from '../ui/Speech';
import { Glitch } from '../robot/Glitch';
import { Btn } from '../ui/Button';
import { bus } from '../events/bus';
import type { Mood } from '../robot/moods';

interface Props {
  beats: Beat[];
  speed: 'normal' | 'fast' | 'instant';
  integrity: number;
  ticks: boolean[];
  rings: number;
  totalRings: number;
  reducedMotion: boolean;
  onBeat?(beat: Beat, index: number): void;
  onDone(): void;
  title?: string;
  kicker?: string;
  showRobotFrom?: number;
  className?: string;
}

export function Cinematic({ beats, speed, integrity, ticks, rings, totalRings, reducedMotion, onBeat, onDone, title, kicker, showRobotFrom = 0, className = '' }: Props) {
  const [index, setIndex] = useState(0);
  const [reply, setReply] = useState<string | null>(null);
  const beat = beats[Math.min(index, beats.length - 1)];
  const text = reply ?? beat.text;
  const { shown, done, finish } = useTypewriter(text, speed, `${index}-${reply ?? ''}`, beat.speaker !== 'narrator');
  const lastMood = beats.slice(0, index + 1).reverse().find(b => b.mood)?.mood ?? 'idle';
  const robotVisible = index >= showRobotFrom && beats.slice(0, index + 1).some(b => b.speaker === 'glitch' || b.mood);
  const awaitingChoice = !!beat.choices && reply === null;

  useEffect(() => { onBeat?.(beat, index); bus.emit('cinematic:beat', { mood: beat.mood, fx: beat.fx }); }, [index]); // eslint-disable-line react-hooks/exhaustive-deps

  // empty beats (pure visual moments) advance on their own
  useEffect(() => {
    if (beat.text === '' && !beat.choices) {
      const id = setTimeout(() => (index >= beats.length - 1 ? onDone() : setIndex(i => i + 1)), reducedMotion ? 600 : 2200);
      return () => clearTimeout(id);
    }
  }, [beat, reducedMotion]); // eslint-disable-line react-hooks/exhaustive-deps

  const advance = useCallback(() => {
    if (!done) { finish(); return; }
    if (awaitingChoice) return;
    if (index >= beats.length - 1) { onDone(); return; }
    setReply(null);
    setIndex(i => i + 1);
  }, [done, finish, awaitingChoice, index, beats.length, onDone]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); onDone(); }
      else if ((e.key === ' ' || e.key === 'Enter') && !(e.target instanceof HTMLButtonElement)) { e.preventDefault(); advance(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [advance, onDone]);

  return (
    <main className={`screen cinematic speaker-${beat.speaker} fx-${beat.fx ?? 'none'} ${className}`} onClick={advance}>
      {(title || kicker) && index === 0 && (
        <header className="cine-title">
          {kicker && <p className="cine-kicker">{kicker}</p>}
          {title && <h2>{title}</h2>}
        </header>
      )}
      {robotVisible && (
        <div className="cine-robot">
          <Glitch mood={(beat.mood ?? lastMood) as Mood} integrity={integrity} ticks={ticks} rings={rings} totalRings={totalRings} reducedMotion={reducedMotion} />
        </div>
      )}
      {text && (
        <div className={`cine-text cine-${beat.speaker}`} aria-live="polite">
          {beat.speaker === 'glitch' && <span className="speech-name">GLITCH</span>}
          {beat.speaker === 'system' && <span className="speech-name">A note on how this works</span>}
          <p><span className="sr-only">{text}</span><span aria-hidden="true">{shown}</span>{!done && <span className="caret" aria-hidden="true" />}</p>
          {awaitingChoice && done && (
            <div className="cine-choices" onClick={e => e.stopPropagation()}>
              {beat.choices!.map(c => (
                <Btn key={c.label} variant="ghost" autoFocus={c === beat.choices![0]} onClick={() => c.reply ? setReply(c.reply) : advance()}>{c.label}</Btn>
              ))}
            </div>
          )}
          {done && !awaitingChoice && <span className="cine-next" aria-hidden="true">{index >= beats.length - 1 ? 'continue' : 'next'}</span>}
        </div>
      )}
      <div className="cine-controls" onClick={e => e.stopPropagation()}>
        <span className="cine-progress" aria-label={`Beat ${index + 1} of ${beats.length}`}>
          {beats.map((_, i) => <i key={i} className={i <= index ? 'on' : ''} />)}
        </span>
        <Btn variant="quiet" sound="back" onClick={onDone}>Skip scene</Btn>
      </div>
    </main>
  );
}
