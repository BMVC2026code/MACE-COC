"""Answer normalisation and JSON parsing utilities."""

import json
import re

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def normalize_yes_no(value):
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return "YES"
    if text in {"no", "n", "false", "0"}:
        return "NO"
    match = re.match(r"^(yes|no)\b", text)
    return match.group(1).upper() if match else None


def normalize_short_answer(value):
    text = str(value).strip().lower()
    text = text.replace("x ray", "x-ray").replace("xray", "x-ray")
    text = re.sub(r"\bct scan\b", "ct", text)
    text = re.sub(r"\bmri scan\b", "mri", text)
    text = re.sub(r"\bx-ray scan\b", "x-ray", text)
    text = re.sub(r"[^a-z0-9,+/-]+", " ", text)
    tokens = [NUMBER_WORDS.get(t, t) for t in text.split()]
    tokens = [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokens]
    return " ".join(tokens).strip()


def categorical_correct(prediction, ground_truth):
    pred = normalize_short_answer(prediction)
    gt = normalize_short_answer(ground_truth)
    if not pred or not gt:
        return False
    if pred == gt:
        return True
    gt_parts = [normalize_short_answer(p) for p in re.split(r"[,/;]|\band\b", str(ground_truth), flags=re.I)]
    gt_parts = [p for p in gt_parts if p]
    return pred in gt_parts or any(pred == p or pred in p.split() for p in gt_parts)


def parse_confidence(value, default=0.5):
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return default
    if 1.0 < conf <= 10.0:
        conf /= 10.0
    return max(0.0, min(1.0, conf))


def parse_json_object(text):
    if not text:
        return None
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(stripped[start:end + 1])
    except json.JSONDecodeError:
        return None
