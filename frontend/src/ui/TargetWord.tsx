/**
 * The target word, set huge. While the memory is lost, the word is sliced and
 * jittering; when the CNN recognises the drawing, the slices lock together.
 */
export function TargetWord({ word, state }: { word: string; state: 'lost' | 'scanning' | 'restored' }) {
  return (
    <h1 className={`target-word is-${state}`} aria-label={`Draw: ${word}`} data-text={word} style={{ ['--len' as any]: Math.max(4, word.length) }}>
      <span className="tw-base" aria-hidden="true">{word}</span>
      <span className="tw-slice tw-a" aria-hidden="true">{word}</span>
      <span className="tw-slice tw-b" aria-hidden="true">{word}</span>
    </h1>
  );
}
