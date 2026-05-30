# MACE-CoC

**Multi-Agent Confidence Estimation via Chain-of-Confidence** for medical Visual Question Answering.

MACE-CoC audits each VLM-generated answer through four specialised agents (blind evidence, falsification, alternative-answer pressure, grounding verification) and integrates their reports via a meta-adjudicator into a calibrated confidence score. The predicted answer is never revised — all changes reflect confidence quality alone.

## Hardware

- Minimum: 1× GPU with 48 GB VRAM (e.g., NVIDIA RTX A6000)
- Recommended: 2× GPUs — one for the audit backbone (Qwen2.5-VL-7B), one for the meta-adjudicator (Llama-3-8B)

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate answers (runs one model on one dataset)
python -m generators.llava_next --dataset vqarad --output outputs/generators/llava_next_vqarad.json --device cuda:0

# 3. Run MACE-CoC audit
python -m mace_coc.audit_runner \
    --dataset vqarad \
    --generator-json outputs/generators/llava_next_vqarad.json \
    --answer-key answer --confidence-key confidence --reasoning-key reasoning \
    --output outputs/mace_coc/llava_next_vqarad.jsonl \
    --audit-device cuda:0 --meta-device cuda:1

# 4. Evaluate
python -m evaluation.score outputs/mace_coc/llava_next_vqarad.jsonl
```

## Full Reproduction

```bash
# Run everything (all 4 models × 3 datasets)
bash scripts/run_all.sh
```

## Scripts

| Script | What it does |
|--------|-------------|
| `scripts/run_generators.sh` | Runs all 4 generators on all 3 datasets |
| `scripts/run_mace_coc.sh` | Runs MACE-CoC audit on all generator outputs |
| `scripts/evaluate.sh` | Scores all outputs (AUROC, ECE, HCE) |
| `scripts/run_all.sh` | Full pipeline end-to-end |

## Models

| Model | Role | HuggingFace ID |
|-------|------|----------------|
| LLaVA-NeXT | Generator | `llava-hf/llava-v1.6-mistral-7b-hf` |
| LLaVA-Med | Generator | `chaoyinshe/llava-med-v1.5-mistral-7b-hf` |
| Hulu-Med | Generator | `ZJU-AI4H/Hulu-Med-7B` |
| MedGemma | Generator | `google/medgemma-4b-it` |
| Qwen2.5-VL-7B | Audit backbone | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Llama-3-8B | Meta-adjudicator | `meta-llama/Meta-Llama-3-8B-Instruct` |

## Datasets

- **VQA-RAD**: Auto-downloaded from HuggingFace (`flaviagiammarino/vqa-rad`)
- **SLAKE**: Auto-downloaded from HuggingFace (`mdwiratathya/SLAKE-vqa-english`)
- **MIMIC-CXR-VQA**: Requires local setup — place subset JSON in `data/mimic_cxr_vqa/`

## Citation

```bibtex
@inproceedings{mace-coc2026,

}
```
