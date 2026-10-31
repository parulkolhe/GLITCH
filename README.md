# GLITCH: A Memory Recovery Protocol

A small robot called GLITCH has lost the link between words and shapes. It still
knows the word *star*; it no longer knows what a star looks like. You draw, freely
and in your own way, and a convolutional neural network trained on Quick, Draw!
sketches decides what it sees. When it sees the word it was given, a memory comes back.

* Three acts, six memory zones, 32 drawable memories, an archive of your own drawings,
  a finale built out of everything you drew.
* No tracing, ever: the canvas starts blank. Hints (“echoes”) are words, not pictures.
* Real recognition: every attempt runs through the trained CNN. Success means the
  network’s top-1 guess is the target, nothing else.
* Honest machine learning: calibrated scores, a validation-fitted “I’m not sure”
  threshold, the exact 28×28 image the network saw and where it looked, and an in-game
  model card. Drawing never trains the model, and the game says so.
* Everything you see and hear is made in code: an SVG robot, a Canvas 2D world, Web Audio
  synthesis and a procedural soundtrack. No image, audio or font files.

## Play

```bash
./play.sh                  # macOS / Linux: creates .venv on first run, opens the game
# or double-click PLAY.command in Finder (macOS)
# or: powershell -ExecutionPolicy Bypass -File play.ps1   (Windows)
```

Manual start, if you prefer:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt          # FastAPI, NumPy, Pillow. No TensorFlow.
uvicorn backend.main:app --host 127.0.0.1 --port 8001
open http://127.0.0.1:8001
```

The built game is checked in (`frontend/dist`), so Node is only needed to change the frontend.
Sound starts after your first key press or tap (browser autoplay rules). Controls: draw with
a mouse, finger or stylus; **Enter** transmits, **Backspace** undoes, **Delete** clears,
**Esc** pauses or skips a cutscene. Settings cover volume (master, music, effects), mute,
reduced motion, text speed and whether to show the network’s view after each drawing.

## How the model is loaded

The server loads the CNN once at startup, in this order:

1. `backend/model/registry.json` → `backend/model/versions/<active>/` (weights, calibration,
   uncertainty policy, difficulty tiers, test metrics);
2. `backend/model/doodle_classifier.weights.npz` + `.arch.json`;
3. `backend/model/doodle_classifier.keras`, exported automatically to (2).

Inference runs on NumPy (`backend/ml/engine.py`), a faithful replay of the Keras graph. It
reproduces the checkpoint’s recorded held-out test accuracy exactly (0.87359375 on 32,000
drawings) and its saved probabilities to within 2.4e-4 (float16 rounding).
Check it yourself: `python ml/verify_engine.py`.

## Frontend development

```bash
cd frontend
npm install
npm run build        # esbuild → frontend/dist (served by the backend at /)
npm run watch        # rebuild on change; keep uvicorn running and reload the page
npm run typecheck    # tsc with @types/react
npm test             # state machine, campaign, scoring, saves, strokes, dialogue, API client
```

## Training and evaluating the CNN

Needs `pip install -r ml/requirements.txt` (TensorFlow) and the prepared splits in
`data/quickdraw/processed/` (`python ml/prepare_data.py --download` builds them:
32 classes × 10,000 drawings, 8,000 / 1,000 / 1,000 per class, seed 42).

```bash
python ml/train.py --epochs 30 --batch-size 256 --patience 4     # the original baseline recipe
python ml/experiments.py                                         # controlled architecture comparison (validation only)
python ml/export_weights.py --model experiments/compact_bn_cnn/model.keras --out-prefix experiments/compact_bn_cnn/model
python ml/promote.py --candidate experiments/compact_bn_cnn --version compact-v2 --activate   # test once, then serve
python ml/promote.py --activate-version baseline-v1              # roll back
```

`ml/promote.py` keeps every version, fits temperature and the uncertainty threshold on the
validation split, scores the test split once, and refuses to activate a candidate whose
validation accuracy is lower than the active model’s unless forced.

Analysis scripts (NumPy only, run anywhere):

| Script | Question it answers | Output |
|---|---|---|
| `ml/verify_engine.py` | Does the NumPy engine match Keras? | prints parity |
| `ml/pipeline_parity.py` | Does the game’s canvas path preserve the training representation? | `reports/pipeline_parity.json` |
| `ml/tune_preprocessing.py` | Which crop margin / stroke dilation make canvas drawings look like training data? | `reports/preprocessing_tuning.json` |
| `ml/robustness.py` | How does accuracy hold up under shifts, rotation, scale, stroke width? | `reports/robustness.json` |
| `ml/eval_canvas.py` | Accuracy on real drawings collected at `/collect` | `reports/canvas_eval.json` |
| `ml/plots.py` | Figures | `reports/figures/` |

Results, method and limitations: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Tests

```bash
python -m unittest backend.tests.test_predict backend.tests.test_explain ml.tests.test_glitch_ml ml.tests.test_data_pipeline
cd frontend && npm test
```

The API tests build a tiny model in the exported format, so they need neither TensorFlow nor
the trained weights; tests marked for the trained model run when it is present. TensorFlow-only
tests (the training graph) skip when TensorFlow is not installed.

## Project layout

```text
backend/            FastAPI server, preprocessing, NumPy engine, model registry
  model/            class names, active model versions, registry.json
frontend/           the game: React + TypeScript, built with esbuild
  src/game/         phase machine, campaign, scoring, saves, controller hook
  src/story/        all dialogue, memories and echoes (hand-written)
  src/robot/        GLITCH: SVG body + mood/pose system
  src/fx/           procedural world canvas and one-shot effects
  src/audio/        synthesiser, procedural music, event→sound wiring
  src/screens/      boot, title, cinematics, core activation, play, archive, finale
ml/                 data prep, training, experiments, evaluation, promotion, analysis
docs/               architecture, model card
reports/            metrics, figures and analysis outputs
legacy/doodle-panic the previous UI (Doodle Panic!!), kept for reference
```

Architecture, phase machine, API contract and event system: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Known issues

* The network is confidently wrong on scribbles that resemble nothing: softmax scores are
  calibrated on Quick, Draw! drawings, not on random marks. A four-line zig-zag can score 96%
  “lightning”. The uncertainty threshold catches genuinely ambiguous drawings, not
  out-of-distribution ones. Possible fix: an explicit rejection model or energy-based OOD score.
* Animals (bird/duck, cat/dog, frog/dog) are the weakest classes; the game warns you, and
  GLITCH’s lines for those confusions come from the real confusion matrix.
* Saves live in one browser’s local storage only.
* Without `@types/react` installed, `npm run typecheck:offline` type-checks against a small
  local React shim instead (used where npm is unavailable).
