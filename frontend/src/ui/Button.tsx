/** Buttons that speak the game's sound language. */
import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { bus } from '../events/bus';

type Variant = 'primary' | 'ghost' | 'quiet' | 'icon';

export function Btn({ variant = 'ghost', sound = 'soft', children, onClick, className = '', ...rest }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; sound?: 'soft' | 'confirm' | 'back' | 'none'; children: ReactNode }) {
  return (
    <button
      type="button"
      className={`btn btn-${variant} ${className}`}
      onPointerEnter={() => bus.emit('ui:hover', {})}
      onClick={e => { if (sound !== 'none') bus.emit('ui:click', { variant: sound }); onClick?.(e); }}
      {...rest}
    >
      {children}
    </button>
  );
}
