#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$DIR/submit_experiment.sh" --exp lambda_regression "$@"
