# GLITCH architecture

```text
┌──────────────────────── browser (frontend/, React + TypeScript) ─────────────────────────┐
│                                                                                          │
│  screens/*  ──read state / call actions──►  game/useGame.tsx  (controller)               │
│     ▲                                         │  • phase machine   game/stateMachine.ts  │
│     │                                         │  • campaign        game/campaign.ts      │
│  robot/Glitch.tsx  fx/World.tsx  ui/*         │  • scoring         game/scoring.ts       │
│     ▲      ▲                                  │  • progress/save   game/progress.ts, save│
│     │      │          events/bus.ts  ◄──emit──┤  • achievements    game/achievements.ts  │
│     │      └── fx/fxController ◄──────────────┤  • dialogue        story/*               │
│  audio/wire.ts ──► audio/engine.ts, music.ts  │                                          │
│                                               ▼                                          │
│  canvas/DrawingCanvas.tsx ── strokes ──► canvas/strokes.ts exportPng (black on white)    │
│                                               │                                          │
│                              services/predictionClient.ts  (request ids, abort, timeout) │
└───────────────────────────────────────────────┼──────────────────────────────────────────┘
                                                │ POST /api/predict {image_base64, target, request_id}
┌─────────────────────────── backend/ (FastAPI) ▼──────────────────────────────────────────┐
│  api/routes.py      validation, error codes (422 bad drawing, 503 no model, 413 too big) │
│  ml/preprocessing.py  PNG → ink mask → crop/centre/dilate → 28×28 uint8 (QuickDraw form) │
│  services/prediction.py  one forward pass → temperature scaling → top-5, uncertainty,    │
│                          Grad-CAM, the exact 28×28 input, latency                        │
│  ml/engine.py       NumPy replay of the Keras graph (no TensorFlow needed to play)       │
│  ml/model.py        loads registry.json → versions/<active>/ (weights, calibration,      │
│                     confidence policy, difficulty tiers); model loaded once at startup   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## The phase machine

Every screen is a phase. Every change of phase is an event with a row in `TRANSITIONS`
(`frontend/src/game/stateMachine.ts`); anything else is rejected and counted.

```text
BOOT ─BOOT_DONE→ TITLE ─NEW_GAME→ INTRO_CINEMATIC ─INTRO_DONE→ MEMORY_CORE_ACTIVATION
TITLE ─CONTINUE→ ZONE_INTRO            MEMORY_CORE_ACTIVATION ─CORE_ONLINE→ ZONE_INTRO
TITLE ─PRACTICE→ DRAWING_READY  (free drawing, signal rush, revisiting dark memories)
ZONE_INTRO ─ZONE_READY→ DRAWING_READY ─STROKE_START→ DRAWING_IN_PROGRESS ─SUBMIT→ PREDICTION_PENDING
PREDICTION_PENDING ─RESULT_RECOGNIZED→ PREDICTION_RECOGNIZED ─RESTORE→ MEMORY_RESTORATION
PREDICTION_PENDING ─RESULT_MISSED / RESULT_UNCERTAIN / RESULT_ERROR→ retry phases
retry phases ─STROKE_START / SUBMIT / CLEAR / SKIP_TARGET→ …
PREDICTION_PENDING ─CLEAR→ DRAWING_READY   (request aborted; a late answer is discarded)
MEMORY_RESTORATION ─NEXT_TARGET→ DRAWING_READY | ─ZONE_DONE→ LEVEL_COMPLETE
LEVEL_COMPLETE ─NEXT_ZONE→ ZONE_INTRO | ─BEGIN_FINALE→ FINAL_SEQUENCE ─FINALE_DONE→ CAMPAIGN_COMPLETE
any phase except BOOT ─TO_TITLE→ TITLE
```

Guarantees, each covered by `frontend/tests/logic.test.ts`:

* a miss or an uncertain result can never reach `MEMORY_RESTORATION`;
* a prediction cannot be submitted twice, or from an empty canvas;
* a response that arrives after the player cleared, skipped or left is dropped
  (phase serial + target check in `useGame`, request id + abort in `PredictionClient`).

## The success rule

A memory is restored **only** when the CNN's top-1 class equals the target. The calibrated
score shapes what GLITCH says, when echoes appear and the clarity bonus, never success.
Uncertainty (top-1 below the validation-fitted threshold) changes the dialogue to “I’m not
sure”, and it is still a miss.

## Events, sound and visuals

Gameplay code emits facts on `events/bus.ts` (`predict:start`, `memory:recognized`,
`zone:complete`, …). `audio/wire.ts` is the only place that maps events to sounds; the
world layer and the robot react to the same events. Nothing plays on a timer for its own sake.

* **Audio** (`audio/engine.ts`): Web Audio voices → SFX bus / music bus → master →
  compressor-limiter; a procedurally generated convolution reverb on a send; per-sound
  cooldowns and a voice cap; the context is created only after a user gesture.
* **Music** (`audio/music.ts`): a lookahead sequencer; zone sets scale/root/tempo,
  campaign corruption closes the filter and adds bit-noise, the recognition streak adds layers.
* **World** (`fx/World.tsx`): one Canvas 2D loop; sky, stars, scene (Atlas icons, signal
  flood, shatter, constellations of the player's own recovered drawings), horizon grid styled
  per zone, particles, pulses, corruption tears. Reduced motion throttles it to near-static.
* **Robot** (`robot/Glitch.tsx`): SVG; mood → pose table (`robot/moods.ts`); one rAF loop for
  bob, blink, antenna sway (droops with damage), gaze toward the pen; memory-core ticks = real
  recovered classes, rings = completed zones.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | the game (built `frontend/dist`) |
| GET | `/health` | liveness, model readiness, version |
| GET | `/health/ready` | 200 only when the model is loaded, else 503 |
| GET | `/api/model` | version, architecture, parameters, classes, difficulty tiers, temperature, uncertainty threshold, test metrics, mean latency |
| POST | `/api/predict` | `{image_base64, target?, request_id?}` → ranked scores, `uncertain`, entropy, margin, target rank/score, Grad-CAM map, the 28×28 model input, version, `inference_ms`, echoed `request_id` |
| GET | `/classes`, `/game/meta`; POST `/predict` | kept for the older Doodle Panic client and `collect.html` |
| GET | `/collect` | drawing-collection page for `ml/eval_canvas.py` |
| GET | `/docs` | OpenAPI UI |

Errors: 422 (not a PNG, blank drawing, unknown target), 413 (body over 3.5 MB),
503 (model not loaded), 500 (inference failed). The game shows a retry message for every
error and never substitutes a prediction.

## Persistence

`localStorage` key `glitch.campaign.v2` (validated field by field on load; a corrupt or
foreign save is ignored, never trusted) and `glitch.settings.v1`. Recovered drawings are
stored as simplified stroke lists (Ramer–Douglas–Peucker), a few hundred bytes each. Practice
and rush modes never write campaign progress.
