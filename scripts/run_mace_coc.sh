#!/bin/bash
# Run MACE-CoC audit on all generator outputs
# Usage: bash scripts/run_mace_coc.sh [AUDIT_DEVICE] [META_DEVICE]
set -e

AUDIT_DEVICE=${1:-cuda:0}
META_DEVICE=${2:-cuda:1}
GEN=outputs/generators
OUT=outputs/mace_coc

run_audit() {
    local MODEL=$1 DATASET=$2 ANS_KEY=$3 CONF_KEY=$4 REASON_KEY=$5
    local EXTRA=""
    if [ "$DATASET" = "mimic_cxr_vqa_1k" ]; then
        EXTRA="--subset-json data/mimic_cxr_vqa/test_close_verify_balanced_1000.json"
    fi
    python -m mace_coc.audit_runner \
        --dataset $DATASET \
        --generator-json $GEN/${MODEL}_${DATASET/mimic_cxr_vqa_1k/mimic}.json \
        --answer-key $ANS_KEY --confidence-key $CONF_KEY --reasoning-key $REASON_KEY \
        --output $OUT/${MODEL}_${DATASET}.jsonl \
        --audit-device $AUDIT_DEVICE --meta-device $META_DEVICE \
        $EXTRA
}

# All generators use consistent keys: answer, confidence, reasoning
for MODEL in llava_next llava_med hulu_med medgemma; do
    for DATASET in vqarad slake mimic_cxr_vqa_1k; do
        echo "=== $MODEL / $DATASET ==="
        run_audit $MODEL $DATASET answer confidence reasoning
    done
done

echo "All MACE-CoC audits done."
