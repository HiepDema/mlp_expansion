#!/bin/bash
# Make the official E2E NLG Challenge scorer (./e2e-metrics) runnable and prove it works.
#   bash setup_metrics.sh            (needs git, Java 1.8+, and the same python as training)
# Clones tuetschek/e2e-metrics if ./e2e-metrics is missing, installs its Python deps,
# then scores a two-sentence example end to end (PTB tokenizer and METEOR exercise Java).
set -eu
cd "$(dirname "$0")"
PY="${PY:-python}"

[ -d e2e-metrics ] || git clone --depth 1 https://github.com/tuetschek/e2e-metrics
"$PY" -m pip install -q future matplotlib scikit-image
"${JAVA_HOME:+$JAVA_HOME/bin/}java" -version
"$PY" -c "from mlpx.official import check; print('official scorer OK:', check())"
