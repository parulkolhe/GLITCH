/**
 * What GLITCH actually saw. Shows the model's real output for the last
 * drawing: the ranked top-5 scores, the exact 28x28 bitmap the CNN received
 * after preprocessing, and a Grad-CAM map of where the top guess came from.
 */
import { useEffect, useRef } from 'react';
import type { PredictionResult } from '../services/predictionClient';

function AttentionView({ image, attention }: { image: string; attention: number[][] | null }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const ctx = el.getContext('2d')!;
    const img = new Image();
    img.onload = () => {
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, el.width, el.height);
      ctx.drawImage(img, 0, 0, el.width, el.height);
      if (attention && attention.length) {
        const rows = attention.length, cols = attention[0].length;
        const cw = el.width / cols, ch = el.height / rows;
        for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
          const v = attention[r][c];
          if (v <= 0.05) continue;
          ctx.fillStyle = `rgba(200,255,61,${v * 0.55})`;
          ctx.fillRect(c * cw, r * ch, cw, ch);
        }
      }
    };
    img.src = image;
  }, [image, attention]);
  return <canvas ref={canvas} width={168} height={168} className="attention-canvas" />;
}

export function ScanReadout({ result, target, compact = false }: { result: PredictionResult; target: string; compact?: boolean }) {
  const rows = result.predictions.slice(0, compact ? 3 : 5);
  return (
    <section className={`scan-readout${compact ? ' is-compact' : ''}`} aria-label="Model output for your drawing">
      <div className="sr-view">
        <AttentionView image={result.modelInput} attention={result.attention} />
        <p className="sr-caption">The 28×28 image the network saw{result.attention ? '; green marks where its top guess came from' : ''}.</p>
      </div>
      <ol className="sr-bars">
        {rows.map(row => (
          <li key={row.className} className={row.className === target ? 'is-target' : ''}>
            <span className="sr-label">{row.className}</span>
            <span className="sr-track"><span className="sr-fill" style={{ width: `${Math.max(1.5, row.score * 100)}%` }} /></span>
            <span className="sr-value">{(row.score * 100).toFixed(row.score < 0.1 ? 1 : 0)}%</span>
          </li>
        ))}
        {result.targetRank && result.targetRank > rows.length && (
          <li className="is-target is-offlist">
            <span className="sr-label">{target}</span>
            <span className="sr-track"><span className="sr-fill" style={{ width: `${Math.max(1, (result.targetScore ?? 0) * 100)}%` }} /></span>
            <span className="sr-value">#{result.targetRank}</span>
          </li>
        )}
      </ol>
      <p className="sr-meta">
        {result.uncertain ? `Below the ${Math.round(result.uncertainBelow * 100)}% confidence line, so GLITCH reports it as unsure. ` : ''}
        Scores are temperature-calibrated on validation data. {result.inferenceMs.toFixed(1)} ms on {result.modelVersion}.
      </p>
    </section>
  );
}
