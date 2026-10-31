/*
 * Doodle Panic!! adaptive game controllers ("agents").
 *
 * Each controller is a small rule-based policy:
 *   observe (round outcome, calibrated confidence, time left)
 *   -> decide (next prompt tier, next time limit, points, feedback)
 *   -> act on the next round.
 * They are deterministic given their inputs, so they are easy to test and explain.
 *
 * Works in the browser (window.DoodleAgents) and in Node (module.exports) for tests.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.DoodleAgents = api;
})(typeof self !== 'undefined' ? self : this, function () {
  const LEVELS = ['easy', 'medium', 'hard'];
  const TIER_MULTIPLIER = { easy: 1, medium: 1.25, hard: 1.5 };
  const TIME = { base: 30, min: 15, max: 40, step: 5 };
  const CONFIDENT = 0.6;   // calibrated confidence treated as a clear recognition
  const UNSURE = 0.4;      // below this the model is guessing

  // Shared view of how one round went, derived from the model's top-3.
  function outcome(predictions, target) {
    const rows = Array.isArray(predictions) ? predictions : [];
    const wanted = String(target).toLowerCase();
    const rank = rows.findIndex(row => String(row.label).toLowerCase() === wanted);
    const top = rows[0] || { label: '?', confidence: 0 };
    return {
      topOne: rank === 0,
      topThree: rank >= 0 && rank < 3,
      targetConfidence: rank >= 0 ? Number(rows[rank].confidence) : 0,
      topLabel: String(top.label),
      topConfidence: Number(top.confidence) || 0,
    };
  }

  /* 1. Difficulty controller: picks the next prompt from easy / medium / hard
   *    tiers (tiers come from per-class test F1 via /game/meta). Level goes up
   *    after a confident top-1 hit, down after a miss outside the top 3. */
  function createDifficultyAgent(tiers, random = Math.random) {
    const byLevel = { easy: [], medium: [], hard: [] };
    for (const [label, tier] of Object.entries(tiers || {})) {
      (byLevel[tier] || byLevel.medium).push(label);
    }
    let level = 0;
    const used = new Set();
    return {
      get level() { return LEVELS[level]; },
      update(result) {
        if (result.topOne && result.targetConfidence >= CONFIDENT) level = Math.min(LEVELS.length - 1, level + 1);
        else if (!result.topThree) level = Math.max(0, level - 1);
        return LEVELS[level];
      },
      nextTarget() {
        // Prefer the current tier; fall back to the nearest tier with unused prompts.
        const order = [level, level - 1, level + 1, level - 2, level + 2].filter(i => i >= 0 && i < LEVELS.length);
        for (const index of order) {
          const options = byLevel[LEVELS[index]].filter(label => !used.has(label));
          if (options.length) {
            const label = options[Math.floor(random() * options.length) % options.length];
            used.add(label);
            return { label, tier: LEVELS[index] };
          }
        }
        return null;
      },
    };
  }

  /* 2. Timer controller: a confident fast win shrinks the next round's time,
   *    a miss gives more time, within 15-40 seconds. */
  function createTimerAgent() {
    let seconds = TIME.base;
    return {
      get seconds() { return seconds; },
      update(result, secondsLeft) {
        if (result.topOne && result.targetConfidence >= CONFIDENT && secondsLeft >= seconds / 2) seconds -= TIME.step;
        else if (!result.topThree) seconds += TIME.step;
        seconds = Math.max(TIME.min, Math.min(TIME.max, seconds));
        return seconds;
      },
    };
  }

  /* 3. Scoring controller: points scale with calibrated confidence in the
   *    target, prompt difficulty, and time left. Uses the model's own output,
   *    never the browser-personalised scores, so saved corrections can't farm points. */
  function score(result, tier, secondsLeft) {
    const multiplier = TIER_MULTIPLIER[tier] || 1;
    const confidenceFactor = 0.5 + 0.5 * Math.max(0, Math.min(1, result.targetConfidence));
    if (result.topOne) return Math.round(100 * multiplier * confidenceFactor + 2 * Math.max(0, secondsLeft));
    if (result.topThree) return Math.round(40 * multiplier * confidenceFactor);
    return 0;
  }

  // Hints for the model's most frequent confusions (from the test confusion matrix).
  const CONFUSION_HINTS = {
    'bird|duck': 'Ducks read best with a flat bill and a body sitting on water; birds with a pointed beak and spread wings.',
    'cat|dog': 'Cats: pointy triangle ears and long whiskers. Dogs: floppy ears and a longer snout.',
    'banana|moon': 'A banana has a stalk at one end and thicker middle; a moon is a clean, thin crescent.',
    'frog|dog': 'Give the frog big eyes on top of a wide, flat head and bent legs.',
    'cat|rabbit': 'Rabbits need long upright ears; cats have short triangles.',
    'airplane|star': 'Draw a long fuselage with two wings and a tail fin, not a symmetric shape.',
    'moon|clock': 'Keep the moon an open crescent; a full circle looks like a clock.',
  };
  function hintFor(a, b) {
    const key = [String(a).toLowerCase(), String(b).toLowerCase()].sort().join('|');
    return CONFUSION_HINTS[key] || null;
  }

  /* 4. Feedback controller: turns the outcome and confidence into a specific
   *    next-step message instead of a generic "wrong". */
  function feedback(result, target) {
    const pct = value => Math.round(value * 100) + '%';
    if (result.topOne && result.targetConfidence >= CONFIDENT) {
      return { tone: 'great', message: `Clear ${target}: recognised at ${pct(result.targetConfidence)}. Next prompt gets harder.` };
    }
    if (result.topOne) {
      return { tone: 'ok', message: `Recognised, but only at ${pct(result.targetConfidence)}. Exaggerate the ${target}'s most distinctive feature.` };
    }
    const hint = hintFor(target, result.topLabel);
    if (result.topThree) {
      return { tone: 'close', message: `GLITCH leaned towards ${result.topLabel}. ${hint || `Make the ${target} unmistakable with one defining detail.`}` };
    }
    if (result.topConfidence < UNSURE) {
      return { tone: 'unsure', message: `GLITCH was unsure (best guess ${result.topLabel}, ${pct(result.topConfidence)}). Try one bold outline that fills the canvas.` };
    }
    return { tone: 'miss', message: `GLITCH was fairly sure it saw a ${result.topLabel} (${pct(result.topConfidence)}). ${hint || 'Next round is an easier prompt with more time.'}` };
  }

  return { LEVELS, TIME, CONFIDENT, UNSURE, outcome, createDifficultyAgent, createTimerAgent, score, feedback, hintFor };
});
