#!/usr/bin/env bash
# GLITCH: retrain, evaluate, analyse and test everything (macOS / Linux). Needs internet only if the data is missing.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pip install -r backend/requirements.txt -r ml/requirements.txt
[ -f data/quickdraw/processed/train.npz ] || python3 ml/prepare_data.py --download
python3 ml/experiments.py                     # controlled comparison on validation -> experiments/comparison.md
echo "Pick a run from experiments/comparison.md, then e.g.:"
echo "  python3 ml/promote.py --candidate experiments/compact_bn_cnn --version compact-v2 --activate"
python3 ml/verify_engine.py
python3 ml/pipeline_parity.py
python3 ml/robustness.py
python3 ml/plots.py
python3 ml/analysis_figures.py
python3 -m unittest backend.tests.test_predict backend.tests.test_explain ml.tests.test_glitch_ml ml.tests.test_data_pipeline
(cd frontend && npm install && npm run build && npm test)
echo "Done. Reports in reports/, figures in reports/figures/."
