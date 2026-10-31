# Viva prep: Doodle Panic!!

Every answer is grounded in this repository. File references are where to point if the examiner asks to see the code.

## 60-second walkthrough

A browser game asks the player to draw a prompt. The canvas PNG goes to a FastAPI endpoint (`backend/api/routes.py`). That endpoint turns it into a 28×28 Quick, Draw!-style bitmap (`backend/ml/preprocessing.py`) and runs a 3-block CNN (`ml/train.py`, 91,136 parameters) trained on 320,000 drawings across 32 classes. The API returns temperature-calibrated top-3 predictions. Four rule-based controllers in `agents.js` then use the result, the confidence and the time left to set the score, the next prompt's difficulty, the next time limit and a coaching message. On the held-out test set the model gets 86.99% top-1, 95.12% top-3 and macro F1 0.869.

## Data

**Why these 32 classes?** They can be drawn in under 30 seconds, and the set includes deliberately similar pairs (cat/dog, bird/duck, banana/moon) so difficulty varies.

**How was the data split?** Per class: 8,000 train, 1,000 validation, 1,000 test, with seed 42 (`ml/prepare_data.py`, `split_indices`). It's stratified, so every split is perfectly balanced. A test (`ml/tests/test_data_pipeline.py`) checks that the splits are disjoint and reproducible.

**Why is macro recall exactly equal to accuracy?** The test set is balanced (1,000 per class), so the mean of per-class recall equals overall accuracy.

**How did you get 10,000 drawings per class without downloading 30+ GB?** HTTP range requests. The script fetches 10 contiguous blocks from 10 regions of each class file, so the sample spans the whole file.

**Limitation to admit:** each block is contiguous, so drawings within a block may be correlated (similar time period). Also, the bitmap files include drawings the original game failed to recognise, which adds some label noise.

## Preprocessing

**Training input vs game input?** Training uses the dataset's own 28×28 bitmaps: white strokes on black, centred, full-intensity strokes. The canvas is a 600×600 image with dark strokes on white, so `preprocess_png` converts it:
1. Binarise: any pixel darker than 245 counts as ink.
2. Crop to the drawing's bounding box and centre it in a square with a 12% margin.
3. Resize to 256 and dilate the strokes (MaxFilter 9).
4. Area-resize to 28×28, invert, and rescale so the peak is 255.

**Why binarise?** The original version used grayscale intensity, so lime ink reached the model at 31/255 (almost blank) and a 2px brush vanished. Now every colour gives an identical bitmap; a unit test checks this.

**Where is normalisation?** It's a `Rescaling(1/255)` layer inside the model, so inference can't forget it or apply it twice.

**What augmentation, and why?** RandomTranslation ±8%, RandomRotation ±0.045 turns (±16°) and RandomZoom ±8%. Players draw at different positions, sizes and slants. These layers are active only during training; Keras skips them at inference.

## Model

**Walk me through the architecture.**

| Stage | Layers | Output |
|---|---|---|
| Block 1 | Conv 3×3 ×32 → BN → MaxPool | 14×14×32 |
| Block 2 | Conv 3×3 ×64 → BN → MaxPool | 7×7×64 |
| Block 3 | Conv 3×3 ×96 → Global Average Pooling | 96 |
| Head | Dropout 0.3 → Dense 128 ReLU (`embedding`) → Dropout 0.25 → Dense 32 softmax | 32 |

**Parameter count?** 91,136.
- Conv layers: (3·3·in + 1)·out. For example, block 3 is (9·64 + 1)·96 = 55,392.
- Batch normalisation adds 4 per channel, 2 of them non-trainable.

**Why Global Average Pooling and not Flatten?** Flatten would give 7·7·96 = 4,704 features and about 602k weights in the dense layer; GAP gives 96 features and 12k weights. That means fewer parameters, less overfitting and some shift tolerance. The cost: it discards *where* features are, which probably feeds the animal confusions.

**Why BatchNorm?** It stabilises and speeds up training. Note that BN keeps running statistics for inference; in very short training runs those statistics lag. We saw this on a tiny synthetic run (89% in training mode vs 30% in inference mode), and it disappears with enough steps.

**Loss and optimiser?** Sparse categorical cross-entropy, because labels are integers, not one-hot. Adam with learning rate 0.001. ReduceLROnPlateau halves the learning rate when validation loss stalls. EarlyStopping on validation accuracy restores the best weights.

**Why is validation accuracy higher than training accuracy (86.7% vs 83.2%)?** Training accuracy is measured with augmentation and dropout switched on, which makes it harder. Validation runs with both off. This is not data leakage.

**Is it overfitting?** No. Validation loss kept falling (0.77 → 0.46) and was still improving at epoch 6. If anything, the model is undertrained.

## Evaluation

**Which split chose the model, and which reported results?** Validation chose the model: checkpointing, early stopping and comparing candidates (`evaluate.py --split validation`). Test was scored once, at the end. Using test to pick between candidates would leak information.

**Headline numbers:**
- 86.99% top-1 and 95.12% top-3 on 32,000 test drawings
- Macro precision 0.869 and macro F1 0.869
- Random guessing would be 3.1%

**Best and worst classes?**
- Best: house 0.97, umbrella and mountain 0.95
- Worst: bird 0.60, frog 0.61, dog 0.66, duck 0.71, cat 0.76

**Most common mistakes?**
- bird↔duck (341), cat↔dog (176), banana↔moon (164): together 16% of all 4,164 errors
- 29% of errors are one animal mistaken for another, though only 6 of the 32 classes are animals

**Why top-3?** The game gives partial points when the target is in the top 3, and it measures whether the model is "close".

**What is calibration and ECE?** A model is calibrated if, among predictions made with 80% confidence, about 80% are correct.
- Expected calibration error (ECE) bins predictions by confidence and averages |confidence − accuracy|, weighted by the number of predictions in each bin.

**What is temperature scaling?** Divide the logits by one number T, fitted on validation to minimise negative log-likelihood (Guo et al., 2017).
- T > 1 softens over-confident outputs.
- The top-1 prediction never changes, so accuracy is unchanged.
- The model ends in softmax, so `softmax(log p / T)` is exactly equivalent (`ml/metrics.py`, `apply_temperature`).

**Why does calibration matter here?** Confidence decides points and level-ups. An over-confident model would reward lucky guesses.

## Adaptive controllers ("agents")

**Are these really AI agents?** They are rule-based controllers. Each one observes the round outcome (rank, calibrated confidence, time left), decides, and acts on the next round. All of them are deterministic and unit-tested (`tests/agents.test.js`). The learning is in the CNN; the controllers turn its output into game decisions.

**Difficulty controller:** prompts come in easy, medium and hard tiers, the thirds of the per-class test F1 (`class_difficulty.json`).
- A confident (≥ 60%) top-1 hit moves up a tier.
- A miss outside the top 3 moves down a tier.
- Prompts never repeat within a game.

**Timer controller:** starts at 30 s.
- A confident hit with at least half the time left: −5 s.
- A miss: +5 s.
- Always kept between 15 and 40 s.

**Scoring controller:**
- Top-1: 100 × tier multiplier (1 / 1.25 / 1.5) × (0.5 + 0.5·confidence) + 2 × seconds left.
- Top-3: 40 × tier multiplier × (0.5 + 0.5·confidence).

**Feedback controller:** gives a specific hint for known confusion pairs. For example, if a cat is guessed as a dog: "pointy triangle ears, long whiskers".

**What is "Teach GLITCH"?** The player can save a correction. The browser stores the drawing's 128-d embedding (L2-normalised) in IndexedDB.
- Later drawings with cosine similarity ≥ 0.86 to a saved example get a suggested label.
- It never changes points. It used to, which let players inflate their score.
- The 0.86 threshold has not been validated. Admit that.

## Engineering

**How do you know the code works?**
- 26 Python tests: data splits, the model's input pipeline, fine-tuning layers, preprocessing (colour, position and thin-brush invariance), metrics, and every API route against a small stand-in model.
- 6 JavaScript tests for the controllers.
- One further test runs only when the real trained model is present.

**Bugs found and fixed:**
1. Fine-tuning reused `base.layers[2:]`, which re-inserted the augmentation layers. So the "no augmentation" run actually had augmentation, and `--augment-finetuning` applied it twice. Test metrics were unaffected because augmentation is off at inference.
2. Coloured and thin strokes reached the model nearly blank.
3. The README told users to compare candidate models on the test set.

**Why FastAPI instead of Streamlit (the original plan)?** It gave full control of the drawing canvas and game loop, a clean JSON API with input validation, and the model loads once at startup.

## Likely curveballs

- **"What would improve accuracy most?"**
  - Train to convergence first; the curves were still rising.
  - Then a deeper network, or Flatten instead of GAP, for the animal classes.
- **"Why 28×28?"** That's the dataset's published bitmap size. A larger input would need re-rendering from the vector stroke data.
- **"What does the embedding represent?"** The 128 activations before the classifier: a learned feature vector where similar drawings sit close together.
- **"Precision vs recall for bird?"** Precision 0.62, recall 0.58. It is both missed and over-predicted, mostly swapped with duck.
- **"Why softmax and not sigmoid?"** Each drawing has exactly one class, so the classes are mutually exclusive.
