#!/bin/bash
cd "$(dirname "$0")"
echo "GLITCH model experiments - running. Log: experiments/run.log (safe to minimise this window)."
./ml/run_experiments_mac.sh
echo "Done. You can close this window."
