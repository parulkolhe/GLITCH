/** GLITCH's voice: a typewriter line with a soft key tick. Text is always fully readable on demand. */
import { useEffect, useRef, useState } from 'react';
import { audio } from '../audio/engine';

export function useTypewriter(text: string, speed: 'normal' | 'fast' | 'instant', key: unknown, sound = true) {
  const [shown, setShown] = useState(speed === 'instant' ? text.length : 0);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    if (speed === 'instant') { setShown(text.length); return; }
    setShown(0);
    const cps = speed === 'fast' ? 110 : 52;
    let i = 0;
    const id = setInterval(() => {
      i = Math.min(text.length, i + 1);
      setShown(i);
      if (sound && i % 2 === 0 && /\w/.test(text[i - 1] ?? '')) audio.play('typeTick');
      if (i >= text.length) clearInterval(id);
    }, 1000 / cps);
    timer.current = id;
    return () => clearInterval(id);
  }, [text, speed, key, sound]);
  // finishing stops the timer too, otherwise its next tick would rewind the text
  const finish = () => { if (timer.current) clearInterval(timer.current); setShown(text.length); };
  return { shown: text.slice(0, shown), done: shown >= text.length, finish };
}

export function Speech({ text, speed, lineKey, speaker = 'GLITCH', tone = 'glitch', className = '' }: {
  text: string; speed: 'normal' | 'fast' | 'instant'; lineKey: unknown; speaker?: string; tone?: 'glitch' | 'narrator' | 'log' | 'system'; className?: string;
}) {
  const { shown, done, finish } = useTypewriter(text, speed, lineKey, tone !== 'narrator');
  return (
    <div className={`speech speech-${tone} ${className}`} onClick={finish} aria-live="polite">
      {tone === 'glitch' && <span className="speech-name">{speaker}</span>}
      <p>
        <span className="sr-only">{text}</span>
        <span aria-hidden="true">{shown}</span>
        {!done && <span className="caret" aria-hidden="true" />}
      </p>
    </div>
  );
}
