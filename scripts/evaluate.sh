#!/bin/bash
# Score all MACE-CoC outputs
# Usage: bash scripts/evaluate.sh
set -e

python -m evaluation.score outputs/mace_coc/*.jsonl
