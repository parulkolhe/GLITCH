/**
 * Minimal React typings used ONLY when @types/react is not installed
 * (tsconfig.shim.json). With `npm install`, the real types are used instead
 * via tsconfig.json. Hooks are typed precisely; JSX intrinsics are loose.
 */
declare module 'react' {
  export type ReactNode = any;
  export type Key = string | number;
  export type SetStateAction<S> = S | ((prev: S) => S);
  export type Dispatch<A> = (value: A) => void;
  export interface RefObject<T> { current: T | null }
  export interface MutableRefObject<T> { current: T }
  export type Ref<T> = RefObject<T> | ((instance: T | null) => void) | null;
  export type FC<P = {}> = (props: P) => any;
  export type CSSProperties = Record<string, string | number | undefined>;
  export interface SVGProps<T> { [key: string]: any; ref?: Ref<T> }
  export interface ButtonHTMLAttributes<T> { [key: string]: any; ref?: Ref<T>; onClick?: (e: MouseEvent & { currentTarget: T }) => void; disabled?: boolean; className?: string; title?: string }
  export type PointerEvent<T = Element> = globalThis.PointerEvent & { currentTarget: T; nativeEvent: globalThis.PointerEvent };
  export function useState<S>(initial: S | (() => S)): [S, Dispatch<SetStateAction<S>>];
  export function useState<S = undefined>(): [S | undefined, Dispatch<SetStateAction<S | undefined>>];
  export function useReducer<S, A>(reducer: (state: S, action: A) => S, initial: S): [S, Dispatch<A>];
  export function useEffect(effect: () => void | (() => void), deps?: readonly unknown[]): void;
  export function useLayoutEffect(effect: () => void | (() => void), deps?: readonly unknown[]): void;
  export function useMemo<T>(factory: () => T, deps: readonly unknown[]): T;
  export function useCallback<T extends (...args: any[]) => any>(fn: T, deps: readonly unknown[]): T;
  export function useRef<T>(initial: T): MutableRefObject<T>;
  export function useRef<T>(initial: T | null): RefObject<T>;
  export function useRef<T = undefined>(): MutableRefObject<T | undefined>;
  export function useContext<T>(ctx: Context<T>): T;
  export function useImperativeHandle<T>(ref: Ref<T> | undefined, init: () => T, deps?: readonly unknown[]): void;
  export interface Context<T> { Provider: FC<{ value: T; children?: ReactNode }>; Consumer: any }
  export function createContext<T>(value: T): Context<T>;
  export function forwardRef<T, P = {}>(render: (props: P, ref: Ref<T>) => any): FC<P & { ref?: Ref<T> }>;
  const React: { createElement: any };
  export default React;
}
declare module 'react/jsx-runtime' { export const jsx: any; export const jsxs: any; export const Fragment: any; }
declare module 'react-dom/client' { export function createRoot(el: Element): { render(node: any): void; unmount(): void }; }
declare namespace React {
  type ReactNode = any;
  type PointerEvent<T = Element> = globalThis.PointerEvent & { currentTarget: T; nativeEvent: globalThis.PointerEvent };
}
declare namespace JSX {
  interface Element { }
  interface IntrinsicElements { [name: string]: any }
  interface ElementChildrenAttribute { children: {} }
  interface IntrinsicAttributes { key?: string | number }
}
declare module '*.css';
