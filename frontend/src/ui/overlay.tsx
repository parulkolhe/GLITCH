import { createContext, useContext } from 'react';

export type OverlayName = 'archive' | 'settings' | 'inside' | 'menu' | null;
export const OverlayContext = createContext<{ overlay: OverlayName; open(o: OverlayName): void }>({ overlay: null, open: () => {} });
export const useOverlay = () => useContext(OverlayContext);
