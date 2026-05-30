"""LLaVA-NeXT generator for medical VQA."""

import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from mace_coc.datasets import load_dataset_by_name
from mace_coc.utils import parse_confidence

MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

PROMPT = """[INST] <image>
You are answering a close-ended medical visual question.

Question: {question}

Return a concise answer and calibrated confidence that your answer is correct.
Use YES or NO for yes/no questions. Otherwise use the shortest clinically valid answer.

Format exactly:
ANSWER: <short answer>
CONFIDENCE: <0-1>
REASONING: <brief image-grounded reason>
[/INST]"""


def parse_response(text):
    ans_m = re.search(r"ANSWER:\s*(.+?)(?:\n|CONFIDENCE:|REASONING:|$)", text, re.I | re.S)
    conf_m = re.search(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", text, re.I)
    reason_m = re.search(r"REASONING:\s*(.*)", text, re.I | re.S)
    answer = ans_m.group(1).strip() if ans_m else ""
    answer = re.sub(r"\s+", " ", answer).strip(" .")
    yesno = re.match(r"^(yes|no)\b", answer, re.I)
    if yesno:
        answer = yesno.group(1).upper()
    confidence = parse_confidence(conf_m.group(1) if conf_m else 0.5)
    reasoning = reason_m.group(1).strip() if reason_m else text.strip()
    return answer, confidence, reasoning


class LlavaNextGenerator:
    def __init__(self, device="cuda:0", model_id=MODEL_ID):
        self.device = device
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map={"": device}).eval()

    def generate(self, image, question):
        prompt = PROMPT.format(question=question)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=192, do_sample=False, use_cache=True)
        text = self.processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        answer, confidence, reasoning = parse_response(text)
        return {"answer": answer, "confidence": confidence, "reasoning": reasoning, "raw": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["vqarad", "slake", "mimic_cxr_vqa_1k"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--subset-json", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    kwargs = {"split": "test"}
    if args.dataset == "mimic_cxr_vqa_1k":
        kwargs["subset_json"] = args.subset_json
    rows = load_dataset_by_name(args.dataset, **kwargs)
    if args.limit:
        rows = rows[:args.limit]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() and out.read_text().strip() else []
    gen = LlavaNextGenerator(device=args.device)

    for idx, item in enumerate(tqdm(rows, desc=f"LLaVA-NeXT {args.dataset}")):
        if idx < len(results):
            continue
        pred = gen.generate(item["image"], item["question"])
        results.append({"question": item["question"], "gt": item["answer"],
                        "answer": pred["answer"], "confidence": pred["confidence"],
                        "reasoning": pred["reasoning"], "raw": pred["raw"]})
        out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
