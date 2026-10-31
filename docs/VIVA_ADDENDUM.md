# Viva addendum: what changed from Doodle Panic to GLITCH

`VIVA_PREP.md` still describes the data, the baseline CNN and the evaluation method. This
covers the questions the new parts invite. Numbers are from `reports/` and `backend/model/versions/baseline-v1/`.

**Which model does the game use, and how do you know it's the right one?**
`baseline-v1`: the 30-epoch checkpoint from the `doodle-panic 2` run (test top-1 87.36%,
macro-F1 0.874). The server replays it in NumPy; `ml/verify_engine.py` shows the NumPy engine
reproduces the recorded test accuracy exactly (0.87359375 on 32,000 drawings) and the saved
Keras probabilities to within 2.4e-4.

**Why remove TensorFlow from the server?**
A 91k-parameter CNN doesn't need a 500 MB runtime to play. The exporter reads the `.keras`
archive (via TF, h5py or the system libhdf5) into `.weights.npz` + `.arch.json`; the engine
supports every layer the training code produces, including residual Add and separable
convolutions. TensorFlow is still used to train.

**What exactly makes a drawing “recognised”?**
Top-1 class == target. Nothing else. Confidence never turns a wrong class into a success;
it shapes the dialogue, when hints appear, and a small clarity bonus on an already-correct
drawing (`frontend/src/game/useGame.tsx`, `scoring.ts`).

**How is “GLITCH is unsure” decided?**
Top-1 calibrated score < 0.69. The threshold is the largest value at which validation
accuracy below it is under 50% (`ml/confidence.py`); 19% of validation drawings fall below it.
Figure: `reports/figures/risk_coverage.png`.

**What is Grad-CAM doing in a NumPy model with no autodiff?**
The head is GAP → Dense(relu) → Dense. With GAP, ∂logit_c/∂A[i,j,k] is the same at every
position: g_k/(H·W), where g = W1 · (W2[:,c] ⊙ 1[pre-activation > 0]). So Grad-CAM's channel
weights can be computed exactly in closed form; a unit test checks them against finite
differences (`backend/tests/test_predict.py`).

**Did you check that canvas drawings look like training drawings?**
Yes, and they didn't fully. Upscaling test bitmaps into fake canvas images and sending them
through the API's preprocessing cost 25 points; strokes came out too heavy and the drawing too
small. A fairer simulation (centrelines redrawn with the real 10-px pen) then drove a
validation-only grid search over crop margin and dilation (`ml/tune_preprocessing.py`,
`reports/preprocessing_tuning.json`).

**What did the robustness sweep show?**
Shifts up to 12% and rotations up to 16° barely matter (inside the augmentation range).
Stroke width is the weak point: 1 px thicker → 66.6%, 1 px thinner → 12.7%. That's why
preprocessing dilation matters, and why `ml/experiments.py` has a stroke-width augmentation arm.

**Does the model learn while people play?**
No, and the game says so in the core-activation scene, the model card screen and the
finale. Updating weights from unvetted player drawings without evaluation would be unsafe
and would break the reproducibility of every number above. The infrastructure for doing it
properly exists: versioned models in `backend/model/versions/`, `ml/promote.py` evaluates a
candidate on fixed splits, refuses a regression, and keeps rollback one command away.

**How is the frontend kept from advancing on a stale prediction?**
Two guards. The client gives each request an id and aborts the previous one; the controller
also checks that the phase serial and target are unchanged before applying a result. Both
are tested (`frontend/tests/predictionClient.test.ts`, `logic.test.ts`).

**Why an explicit state machine?**
So impossible states can't occur: you can't restore a memory from a miss, submit twice, or
have a late answer resolve after clearing. Every transition is a table row, and tests walk
the legal paths and probe the illegal ones.
