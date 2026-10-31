// Bundles the game with esbuild into dist/ (served by the FastAPI backend at /).
//   node build.mjs           production build
//   node build.mjs --watch   rebuild on change
//   node build.mjs --tests   compile tests/*.test.ts into dist-tests/ for `node --test`
import { build, context } from 'esbuild';
import { mkdirSync, readFileSync, writeFileSync, readdirSync, rmSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';

const args = new Set(process.argv.slice(2));

if (args.has('--tests')) {
  rmSync('dist-tests', { recursive: true, force: true });
  const entries = readdirSync('tests').filter(f => f.endsWith('.test.ts')).map(f => `tests/${f}`);
  await build({ entryPoints: entries, outdir: 'dist-tests', bundle: true, platform: 'node', format: 'esm',
    outExtension: { '.js': '.mjs' }, target: 'node20', logLevel: 'warning' });
  process.exit(0);
}

const options = {
  entryPoints: ['src/main.tsx'], bundle: true, outdir: 'dist/assets', format: 'esm', target: 'es2020',
  jsx: 'automatic', minify: !args.has('--watch'), sourcemap: args.has('--watch') ? 'inline' : false, legalComments: 'none',
  define: { 'process.env.NODE_ENV': args.has('--watch') ? '"development"' : '"production"' },
  logLevel: 'info', entryNames: 'game',
};

function writeHtml() {
  const js = readFileSync('dist/assets/game.js');
  const css = (() => { try { return readFileSync('dist/assets/game.css'); } catch { return Buffer.from(''); } })();
  const v = createHash('sha256').update(js).update(css).digest('hex').slice(0, 10);
  const html = readFileSync('index.html', 'utf8').replaceAll('%VERSION%', v);
  writeFileSync('dist/index.html', html);
}

mkdirSync('dist/assets', { recursive: true });
if (args.has('--watch')) {
  const ctx = await context({ ...options, plugins: [{ name: 'html', setup(b) { b.onEnd(r => { if (!r.errors.length) writeHtml(); }); } }] });
  await ctx.watch();
} else {
  await build(options);
  writeHtml();
  const size = statSync('dist/assets/game.js').size;
  console.log(`dist/assets/game.js ${(size / 1024).toFixed(1)} kB`);
}
