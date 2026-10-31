#!/bin/bash
# Controlled architecture + augmentation comparison, using the TensorFlow venv from ../doodle-panic 2
# (override with GLITCH_PYTHON=/path/to/python). Output: experiments/run.log, experiments/comparison.md
cd "$(dirname "$0")/.."
PY="${GLITCH_PYTHON:-../doodle-panic 2/venv/bin/python}"
mkdir -p experiments
{
  echo "started $(date)"
  "$PY" -c "import tensorflow as tf; print('TensorFlow', tf.__version__)" || exit 1
  # 1) two-epoch smoke test on 5% of the data, covering every layer type and the stroke-width layer
  "$PY" ml/experiments.py --smoke --runs baseline_cnn:stroke residual_cnn separable_cnn || { echo "SMOKE FAILED"; exit 1; }
  rm -rf experiments/comparison.* experiments/baseline_cnn_stroke experiments/residual_cnn experiments/separable_cnn
  # 2) the real runs, most informative first (results are saved after each run)
  "$PY" ml/experiments.py --runs baseline_cnn baseline_cnn:stroke compact_bn_cnn compact_bn_cnn:stroke residual_cnn separable_cnn compact_bn_cnn:noaug
  echo "finished $(date) exit=$?"
} >> experiments/run.log 2>&1
