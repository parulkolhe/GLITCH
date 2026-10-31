# Model card: GLITCH recognition network

## Summary

| | |
|---|---|
| Task | Single-label classification of hand-drawn sketches into 32 classes |
| Deployed version | `baseline-v1` (`backend/model/registry.json`) |
| Architecture | `quickdraw_cnn`: 3 convolutional blocks + global average pooling + 128-unit dense + softmax |
| Parameters | 91,136 |
| Input | 28×28×1 uint8, white strokes on black (Quick, Draw! bitmap format); `Rescaling(1/255)` inside the model |
| Output | 32 softmax scores, temperature-scaled at serving time (T = 0.9, fitted on validation) |
| Framework | Trained with TensorFlow/Keras 3.15 (TF 2.20); served with a NumPy replay of the graph (`backend/ml/engine.py`) |
| Held-out test | **top-1 87.36%**, top-3 95.19%, macro-F1 0.874 (32,000 drawings) |
| Calibration | ECE 2.61% raw → **1.25%** after temperature scaling |
| Serving latency | ~2–5 ms per drawing on a laptop CPU (NumPy), plus ~40 ms preprocessing |

## Intended use

Recognising a player's free drawing in GLITCH, a single-player browser game, and giving
the player honest feedback: the ranked guesses, a calibrated score, an explicit “unsure”
state, and where the network looked. It is not intended for any decision about people, for
handwriting/character recognition, or for drawings outside the 32 classes.

**The game never trains the model.** Drawing runs inference only; the weights are fixed.
Model changes happen offline through `ml/experiments.py` → `ml/promote.py`, which keeps every
version, evaluates on fixed splits and keeps the previous model for rollback.

## Data

* Source: Google Quick, Draw! NumPy bitmap dataset (28×28 grayscale renders of player
  drawings collected by the Quick, Draw! game), CC BY 4.0.
* 32 classes: cat, dog, fish, bird, house, tree, flower, sun, moon, star, car, bicycle,
  airplane, sailboat, apple, banana, pizza, cake, umbrella, clock, guitar, chair, key,
  lightning, cloud, headphones, frog, duck, rabbit, butterfly, mountain, book.
* 10,000 drawings per class sampled with HTTP range requests from 10 regions of each class
  file (`ml/prepare_data.py`), split per class 8,000 / 1,000 / 1,000 (train / validation /
  test), seed 42. Perfectly balanced; splits are disjoint by index.
* Known data issues: drawings within one sampled block may be correlated in time; the
  bitmap files include drawings the original game did not recognise (label noise); possible
  exact duplicates across splits were not removed.

## Training (baseline-v1)

Adam (lr 1e-3, halved on validation-loss plateau), sparse categorical cross-entropy, batch
256, 30 epochs, early stopping on validation accuracy with best-weight restore. Training-only
augmentation: random translation ±8%, rotation ±0.045 turns (±16°), zoom ±8%. Best
validation accuracy 87.83% (epoch 29); the curve was still rising slowly, so the model is
slightly under-trained rather than over-fitted.

## Evaluation

All selection uses the validation split; the test split was scored once per final model.

| Split | Top-1 | Top-3 | Macro-F1 |
|---|---:|---:|---:|
| Validation | 87.83% | 95.64% | 0.879 |
| Test | 87.36% | 95.19% | 0.874 |

Weakest classes (test F1): bird 0.604, frog 0.619, dog 0.691, duck 0.708, moon 0.771, rabbit 0.803.
Strongest: house 0.963, umbrella 0.957, sailboat 0.957, headphones 0.951.
Most frequent confusions: bird→duck (16.3% of birds), duck→bird (14.5%), moon→banana (9.1%),
banana→moon (8.8%), cat→dog (7.7%), dog→frog (7.6%). The game's dialogue for misses uses
these real pairs.

Figures: `reports/figures/` (training curves, per-class F1, confusion matrix, top confusions,
reliability diagram, confidence histogram, risk–coverage, robustness, preprocessing grid).

### Confidence and uncertainty

* Temperature T = 0.9 minimises validation NLL (0.4314 → 0.4283). ECE on test: 2.61% raw,
  1.25% scaled. Softmax scores are still not guaranteed probabilities.
* The “unsure” threshold is **0.69**, the largest top-1 score below which validation accuracy
  is under 50% (`ml/confidence.py`). Above it the network answers 81.0% of drawings at
  96.7% accuracy; below it, it is right about half the time. GLITCH says it isn't sure for
  those drawings, and an unsure answer is always a miss for gameplay purposes.

### Robustness (`ml/robustness.py`, 1,920 validation drawings)

| Variation | Accuracy |
|---|---|
| none | 87.97% |
| translation 8% / 12% / 24% of width | 88.4% / 86.8% / 74.0% |
| rotation 16° / 24° / 32° | 86.1% / 80.2% / 67.7% |
| zoom 0.6× / 0.75× / 1.25× | 44.7% / 81.2% / 85.4% |
| stroke 1 px **thinner** | **12.7%** |
| stroke 1 px **thicker** / 2 px thicker | **66.6%** / 31.9% |

Geometric robustness is good inside the trained augmentation range and degrades outside it.
Stroke width is the fragile dimension: the network has only seen Quick, Draw!'s rendering
thickness. This motivates (a) the canvas preprocessing calibration below and (b) the
stroke-width augmentation arm in `ml/experiments.py` (`:stroke`).

### Canvas preprocessing: does the game feed the network what it learned from?

`backend/ml/preprocessing.py` turns a 600×600 canvas PNG into the 28×28 format: binarise,
crop, pad to a centred square, resize to 256, dilate strokes, area-resize to 28, rescale
the peak to 255.

* **Naive round trip** (`ml/pipeline_parity.py`): test bitmaps upscaled, thresholded and sent
  through the API's preprocessing lose 25 points (86.1% → 61.0%). Inputs come out heavier
  (ink coverage 0.31 vs 0.24), more saturated (mean stroke 210 vs 175) and smaller (extent
  0.86 vs 0.93). This simulation exaggerates stroke width (an upscaled 28-px stroke is
  ~45 px at 600, a real pen is 10 px), so it is an upper bound on the gap.
* **Faithful simulation** (`ml/canvas_sim.py`): centrelines of each bitmap (Zhang–Suen
  thinning) are redrawn on a 600-px canvas with the game's 10-px pen at 260–520 px size.
  `ml/tune_preprocessing.py` grid-searches crop margin × stroke dilation on simulated
  validation drawings and scores current vs selected settings once on simulated test
  drawings. Results (1,280 simulated drawings per split, `reports/preprocessing_tuning.json`,
  `reports/figures/preprocessing_tuning.png`):

  | Setting | Validation | Test |
  |---|---:|---:|
  | current: margin 0.12, dilation 9 | 76.5% | 75.4% |
  | best on validation: margin 0.16, dilation 7 | 77.3% | 75.6% |
  | no dilation (dilation 1) | 46–49% | |
  | heavy dilation (17) | 46–54% | |
  | raw test bitmaps, same drawings | | 86.6% |

  **Decision: keep the current settings.** The best grid point improves by 0.8 points on
  validation and 0.2 on test, within one standard error (~1.2 points at n = 1,280). Dilation
  is the parameter that matters (49% → 76%), and the existing value is at the optimum; the
  smaller margin suggested by the naive round trip is wrong once pen width is simulated
  faithfully. The remaining ~11-point gap to raw bitmaps reflects lost detail in the simulation
  and the network's sensitivity to stroke width (see robustness), not crop or dilation
  settings, which is what the `:stroke` training arm targets.

## Architecture comparison

`ml/experiments.py` trains candidates under one protocol (AdamW lr 1e-3, weight decay 1e-4,
plateau LR halving, early stopping, same data, seed and augmentation) and compares them on
validation only: the original 3-block CNN retrained, a compact Conv-BN-ReLU CNN (two convs
per block, 32/64/128), a small residual CNN, a depthwise-separable CNN, an augmentation
ablation and the stroke-width augmentation arm.

**Status: not yet run.** The comparison needs TensorFlow, which is not available in the
environment this version was built in. Run it with `ml/run_experiments_mac.sh` (or
`RUN_EXPERIMENTS.command` on macOS; it uses the TensorFlow venv from the earlier project) or
`python ml/experiments.py`. Results are written to `experiments/comparison.md` after each run.
Promote a winner only through `ml/promote.py`, which re-evaluates it on the fixed splits.
Until then, the deployed model is the original baseline and every number in this card
describes it.

## Limitations

* **Out-of-distribution input**: the network is confidently wrong on random scribbles (e.g. a
  4-segment zig-zag scored 96% “lightning”). Calibration was fitted on Quick, Draw! drawings
  only; the threshold catches ambiguity, not novelty.
* **Animals** are much weaker than objects (F1 0.60–0.80); global average pooling discards
  spatial layout, which likely hurts fine distinctions such as bill vs beak.
* **Domain shift**: players draw with a mouse or finger on a canvas, not in the Quick, Draw!
  app; real-canvas accuracy is lower than test accuracy. Collect labelled drawings at
  `/collect` and run `ml/eval_canvas.py` to measure it; no such dataset is bundled.
* **Cultural bias**: Quick, Draw! drawings reflect how its (largely online) players draw a
  “house” or a “cake”; drawings in other conventions may be misrecognised.
* Thirty-two classes only; anything else is forced into one of them.
