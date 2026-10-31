# GLITCH: retrain, evaluate, analyse and test everything (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File run_all.ps1
$ErrorActionPreference = "Stop"
python -m pip install -r backend/requirements.txt -r ml/requirements.txt
if (-not (Test-Path "data/quickdraw/processed/train.npz")) { python ml/prepare_data.py --download }
python ml/experiments.py
Write-Host "Pick a run from experiments/comparison.md, then: python ml/promote.py --candidate experiments/<run> --version <name> --activate"
python ml/verify_engine.py
python ml/pipeline_parity.py
python ml/robustness.py
python ml/plots.py
python ml/analysis_figures.py
python -m unittest backend.tests.test_predict backend.tests.test_explain ml.tests.test_glitch_ml ml.tests.test_data_pipeline
Push-Location frontend; npm install; npm run build; npm test; Pop-Location
Write-Host "Done. Reports in reports/, figures in reports/figures/."
