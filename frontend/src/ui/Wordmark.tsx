/**
 * The GLITCH wordmark, built from strokes like a vector display: each letter
 * is a polyline that draws itself on, and one letter is permanently "torn".
 */
const LETTERS: Record<string, string> = {
  G: 'M 64 14 L 18 14 L 6 26 L 6 74 L 18 86 L 64 86 L 64 52 L 38 52',
  L: 'M 8 10 L 8 86 L 60 86',
  I: 'M 10 14 L 50 14 M 30 14 L 30 86 M 10 86 L 50 86',
  T: 'M 4 14 L 64 14 M 34 14 L 34 88',
  C: 'M 62 14 L 18 14 L 6 26 L 6 74 L 18 86 L 62 86',
  H: 'M 8 10 L 8 90 M 8 50 L 60 50 M 60 10 L 60 90',
};
const WIDTH: Record<string, number> = { G: 74, L: 66, I: 60, T: 70, C: 70, H: 70 };

export function Wordmark({ className = '', torn = true }: { className?: string; torn?: boolean }) {
  let x = 0;
  const parts = 'GLITCH'.split('').map((ch, i) => {
    const node = (
      <g key={i} transform={`translate(${x} 0)`}>
        <g className={`wm-letter${torn && i === 3 ? ' wm-torn' : ''}`}>
          <path d={LETTERS[ch]} pathLength={1} className="wm-glow" style={{ animationDelay: `${0.12 * i}s` }} />
          <path d={LETTERS[ch]} pathLength={1} className="wm-core" style={{ animationDelay: `${0.12 * i}s` }} />
        </g>
      </g>
    );
    x += WIDTH[ch] + 14;
    return node;
  });
  return (
    <svg className={`wordmark ${className}`} viewBox={`-8 0 ${x + 8} 100`} role="img" aria-label="GLITCH">
      {parts}
    </svg>
  );
}
