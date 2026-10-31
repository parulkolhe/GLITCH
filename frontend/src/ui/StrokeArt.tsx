/** Renders saved compact strokes (the player's own recognised drawing) as SVG. */
import { expandStrokes, simplify } from '../canvas/strokes';

export function StrokeArt({ strokes, className = '', label }: { strokes: number[][]; className?: string; label?: string }) {
  const paths = expandStrokes(strokes).map(s => {
    const pts = simplify(s.points, 0.8);
    if (pts.length === 1) return `M ${pts[0][0]} ${pts[0][1]} l 0.1 0`;
    return pts.map((p, i) => `${i ? 'L' : 'M'} ${p[0].toFixed(0)} ${p[1].toFixed(0)}`).join(' ');
  });
  return (
    <svg className={`stroke-art ${className}`} viewBox="0 0 600 600" role={label ? 'img' : undefined} aria-label={label} aria-hidden={label ? undefined : true}>
      {paths.map((d, i) => <path key={`g${i}`} d={d} className="sa-glow" />)}
      {paths.map((d, i) => <path key={`c${i}`} d={d} className="sa-core" pathLength={1} style={{ animationDelay: `${i * 0.12}s` }} />)}
    </svg>
  );
}
