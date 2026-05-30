"""Dataset loaders for close-ended medical VQA benchmarks."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from PIL import Image

from mace_coc.utils import normalize_short_answer, normalize_yes_no


def load_vqarad(split="test"):
    dataset = load_dataset("flaviagiammarino/vqa-rad", split=split)
    rows = []
    for idx, item in enumerate(dataset):
        answer = normalize_yes_no(item["answer"])
        if answer is None:
            continue
        rows.append({
            "dataset": "vqarad",
            "case_id": f"vqarad-{split}-{idx}",
            "image": item["image"].convert("RGB"),
            "question": item["question"],
            "answer": answer,
        })
    return rows


def load_slake(split="test", metadata_path=None):
    dataset = load_dataset("mdwiratathya/SLAKE-vqa-english", split=split)
    exact_types, question_types = None, None
    if metadata_path and Path(metadata_path).exists():
        metadata = json.loads(Path(metadata_path).read_text())
        exact_types = defaultdict(Counter)
        question_types = defaultdict(Counter)
        for item in metadata:
            if item.get("q_lang") != "en":
                continue
            q = str(item.get("question", "")).strip()
            a = normalize_short_answer(item.get("answer", ""))
            at = item.get("answer_type")
            if q and at:
                exact_types[(q, a)][at] += 1
                question_types[q][at] += 1

    rows = []
    for idx, item in enumerate(dataset):
        if exact_types is not None:
            q = str(item.get("question", "")).strip()
            a = normalize_short_answer(item.get("answer", ""))
            et = exact_types.get((q, a))
            at = et.most_common(1)[0][0] if et else None
            if at is None:
                qt = question_types.get(q)
                at = qt.most_common(1)[0][0] if qt else None
            if at is not None and at != "CLOSED":
                continue
        elif len(str(item["answer"]).strip().split()) > 3:
            continue
        rows.append({
            "dataset": "slake",
            "case_id": f"slake-{split}-{idx}",
            "image": item["image"].convert("RGB"),
            "question": item["question"],
            "answer": normalize_short_answer(item["answer"]),
        })
    return rows


def load_mimic_cxr_vqa(subset_json):
    raw = json.loads(Path(subset_json).read_text())
    rows = []
    for idx, item in enumerate(raw):
        image_path = Path(item["local_image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        rows.append({
            "dataset": "mimic_cxr_vqa_1k",
            "case_id": f"mimic-cxr-vqa-1k-test-{idx}",
            "image": Image.open(image_path).convert("RGB"),
            "question": item["question"],
            "answer": normalize_short_answer(item["answer"]),
        })
    return rows


LOADERS = {
    "vqarad": lambda **kw: load_vqarad(split=kw.get("split", "test")),
    "slake": lambda **kw: load_slake(split=kw.get("split", "test"), metadata_path=kw.get("metadata_path")),
    "mimic_cxr_vqa_1k": lambda **kw: load_mimic_cxr_vqa(kw["subset_json"]),
}


def load_dataset_by_name(name, **kwargs):
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(LOADERS.keys())}")
    return LOADERS[name](**kwargs)
