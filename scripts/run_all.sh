#!/bin/bash
# Full pipeline: generate → audit → evaluate
# Usage: bash scripts/run_all.sh
set -e

echo "Step 1: Running generators..."
bash scripts/run_generators.sh cuda:0

echo "Step 2: Running MACE-CoC audit..."
bash scripts/run_mace_coc.sh cuda:0 cuda:1

echo "Step 3: Evaluating..."
bash scripts/evaluate.sh

echo "Done."
