"""Score MACE-CoC outputs and print AUROC, ECE, HCE."""

import argparse
import json
from pathlib import Path

from mace_coc.utils import categorical_correct, parse_confidence
from mace_coc.metrics import auroc, ece, hce, bootstrap_ci


def load_mace_outputs(path):
    """Load JSONL audit outputs."""
    records = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def score(records):
    labels, initial, final = [], [], []
    for r in records:
        gen = r["generator"]
        correct = 1 if categorical_correct(gen["answer"], r["gt"]) else 0
        labels.append(correct)
        initial.append(float(gen["confidence"]))
        meta = r.get("meta") or {}
        fc = meta.get("final_confidence", gen["confidence"])
        final.append(float(fc))

    n = len(labels)
    acc = sum(labels) / n
    init_auroc = auroc(labels, initial)
    mace_auroc = auroc(labels, final)
    init_ece = ece(labels, initial)
    mace_ece = ece(labels, final)
    init_hce = hce(labels, initial)
    mace_hce = hce(labels, final)
    ci = bootstrap_ci(labels, initial, final)

    return {
        "n": n, "accuracy": acc,
        "initial_auroc": init_auroc, "mace_auroc": mace_auroc,
        "initial_ece": init_ece, "mace_ece": mace_ece,
        "initial_hce": init_hce, "mace_hce": mace_hce,
        "hce_reduction": (init_hce - mace_hce) / max(1, init_hce) * 100,
        "delta_auroc_ci": ci["delta_auroc"],
        "delta_ece_ci": ci["delta_ece"],
    }


def main():
    parser = argparse.ArgumentParser(description="Score MACE-CoC audit outputs")
    parser.add_argument("outputs", nargs="+", help="JSONL audit output files")
    args = parser.parse_args()

    print(f"{'File':<50} {'N':>4} {'Acc':>5} {'AUROC_i':>7} {'AUROC_m':>7} "
          f"{'ECE_i':>6} {'ECE_m':>6} {'HCE_i':>5} {'HCE_m':>5} {'Red%':>5}")
    print("-" * 110)

    for path in args.outputs:
        records = load_mace_outputs(path)
        if not records:
            print(f"{path:<50} (empty)")
            continue
        s = score(records)
        print(f"{Path(path).name:<50} {s['n']:>4} {s['accuracy']:>5.2f} "
              f"{s['initial_auroc']:>7.3f} {s['mace_auroc']:>7.3f} "
              f"{s['initial_ece']:>6.3f} {s['mace_ece']:>6.3f} "
              f"{s['initial_hce']:>5} {s['mace_hce']:>5} "
              f"{s['hce_reduction']:>5.1f}")
        print(f"  ΔAUROC 95% CI: [{s['delta_auroc_ci'][0]:+.3f}, {s['delta_auroc_ci'][1]:+.3f}]  "
              f"ΔECE 95% CI: [{s['delta_ece_ci'][0]:+.3f}, {s['delta_ece_ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
