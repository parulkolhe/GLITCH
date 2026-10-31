/** One consistent inline-SVG icon set: 24px grid, 1.8px round strokes. */
import type { SVGProps } from 'react';

const base = (p: SVGProps<SVGSVGElement>) => ({ width: 22, height: 22, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor',
  strokeWidth: 1.8, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true, ...p });

export const IconUndo = (p: SVGProps<SVGSVGElement>) => <svg {...base(p)}><path d="M9 14 4 9l5-5" /><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11" /></svg>;
export const IconClear = (p: SVGProps<SVGSVGElement>) => <svg {...base(p)}><path d="M4 7h16" /><path d="M9 7V4h6v3" /><path d="m6 7 1 13h10l1-13" /></svg>;
export const IconTransmit = (p: SVGProps<SVGSVGElement>) => <svg {...base(p)}><path d="M12 20V9" /><path d="m7 13 5-5 5 5" /><path d="M5 4.5a10 10 0 0 1 14 0" /></svg>;
export const IconEcho = (p: SVGProps<SVGSVGElement>) => <svg {...base(p)}><circle cx="12" cy="12" r="2" /><path d="M7.5 7.5a6.4 6.4 0 0 0 0 9M16.5 7.5a6.4 6.4 0 0 1 0 9" /><path d="M4.6 4.6a10.5 10.5 0 0 0 0 14.8M19.4 4.6a10.5 10.5 0 0 1 0 14.8" /></svg>;
export const IconSkip = (p: SVGProps<SVGSVGElement>) => <svg {...base(p)}><path d="m5 6 7 6-7 6" /><path d="M13 6l7 6-7 6" opacity=".5" /></svg>;
export const IconArchive = (p: SVGProps<SVGSVGElement>) => <svg {...base(p)}><rect x="3.5" y="4" width="17" height="16" rx="2" /><path d="M3.5 9.5h17M9 13h6" /></svg>;
export const IconSettings = (p: SVGProps<SVGSVGElement>) => <svg {...base(p)}><path d="M4 7h10M18 7h2M4 17h4M12 17h8" /><circle cx="16" cy="7" r="2" /><circle cx="10" cy="17" r="2" /></svg>;
export const IconSound = (p: SVGProps<SVGSVGElement>) => <svg {...base(p)}><path d="M4 9.5h3l5-4v13l-5-4H4z" /><path d="M16 9a4 4 0 0 1 0 6M18.5 6.5a7.5 7.5 0 0 1 0 11" /></svg>;
export const IconMute = (p: SVGProps<SVGSVGElement>) => <svg {...base(p)}><path d="M4 9.5h3l5-4v13l-5-4H4z" /><path d="m16 9.5 5 5M21 9.5l-5 5" /></svg>;
export const IconClose = (p: SVGProps<SVGSVGElement>) => <svg {...base(p)}><path d="M6 6l12 12M18 6 6 18" /></svg>;
export const IconMenu = (p: SVGProps<SVGSVGElement>) => <svg {...base(p)}><path d="M4 7h16M4 12h10M4 17h16" /></svg>;
export const IconBrain = (p: SVGProps<SVGSVGElement>) => <svg {...base(p)}><rect x="4" y="4" width="16" height="16" rx="3" /><path d="M8 8h2v2H8zM14 8h2v2h-2zM11 11h2v2h-2zM8 14h2v2H8zM14 14h2v2h-2z" /></svg>;
