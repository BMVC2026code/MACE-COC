#!/bin/bash
# Run all 4 generators on all 3 datasets (12 jobs)
# Usage: bash scripts/run_generators.sh [DEVICE]
set -e

DEVICE=${1:-cuda:0}
OUT=outputs/generators

# --- LLaVA-NeXT ---
python -m generators.llava_next --dataset vqarad --output $OUT/llava_next_vqarad.json --device $DEVICE
python -m generators.llava_next --dataset slake  --output $OUT/llava_next_slake.json  --device $DEVICE
python -m generators.llava_next --dataset mimic_cxr_vqa_1k --output $OUT/llava_next_mimic.json --device $DEVICE --subset-json data/mimic_cxr_vqa/test_close_verify_balanced_1000.json

# --- LLaVA-Med ---
python -m generators.llava_med --dataset vqarad --output $OUT/llava_med_vqarad.json --device $DEVICE
python -m generators.llava_med --dataset slake  --output $OUT/llava_med_slake.json  --device $DEVICE
python -m generators.llava_med --dataset mimic_cxr_vqa_1k --output $OUT/llava_med_mimic.json --device $DEVICE --subset-json data/mimic_cxr_vqa/test_close_verify_balanced_1000.json

# --- Hulu-Med ---
python -m generators.hulu_med --dataset vqarad --output $OUT/hulu_med_vqarad.json --device $DEVICE
python -m generators.hulu_med --dataset slake  --output $OUT/hulu_med_slake.json  --device $DEVICE
python -m generators.hulu_med --dataset mimic_cxr_vqa_1k --output $OUT/hulu_med_mimic.json --device $DEVICE --subset-json data/mimic_cxr_vqa/test_close_verify_balanced_1000.json

# --- MedGemma ---
python -m generators.medgemma --dataset vqarad --output $OUT/medgemma_vqarad.json --device $DEVICE
python -m generators.medgemma --dataset slake  --output $OUT/medgemma_slake.json  --device $DEVICE
python -m generators.medgemma --dataset mimic_cxr_vqa_1k --output $OUT/medgemma_mimic.json --device $DEVICE --subset-json data/mimic_cxr_vqa/test_close_verify_balanced_1000.json

echo "All generators done."
