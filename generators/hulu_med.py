"""Hulu-Med generator for medical VQA."""

import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from mace_coc.datasets import load_dataset_by_name
from mace_coc.utils import parse_confidence

MODEL_ID = "ZJU-AI4H/Hulu-Med-7B"
MAX_IMAGE_EDGE = 1024

PROMPT = (
    "You are answering a medical visual question about a radiology image.\n\n"
    "Question: {question}\n\n"
    "Return a short final answer and a calibrated confidence that the answer is correct.\n"
    "If the question is yes/no, answer YES or NO.\n"
    "Otherwise, answer with the shortest clinically valid term.\n\n"
    "Format exactly:\nANSWER: <short answer>\nCONFIDENCE: <0-1>\nREASONING: <brief image-grounded reason>\n"
)


def resize_for_hulu(image, max_edge=MAX_IMAGE_EDGE):
    image = image.convert("RGB")
    w, h = image.size
    longest = max(w, h)
    if longest <= max_edge:
        return image
    scale = max_edge / float(longest)
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))))


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


class HuluMedGenerator:
    def __init__(self, device="cuda:0", model_id=MODEL_ID):
        self.device = device
        hulu_proc_cls = get_class_from_dynamic_module(
            "processing_hulumed.HulumedProcessor", model_id, trust_remote_code=True)
        orig_get_args = hulu_proc_cls._get_arguments_from_pretrained.__func__
        @classmethod
        def fixed_get_args(cls, pretrained_model_name_or_path, *args, **kwargs):
            return orig_get_args(cls, pretrained_model_name_or_path, **kwargs)
        hulu_proc_cls._get_arguments_from_pretrained = fixed_get_args
        self.processor = hulu_proc_cls.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, load_in_4bit=True,
            device_map={"": device}).eval()

    def generate(self, image, question):
        image = resize_for_hulu(image)
        prompt = PROMPT.format(question=question)
        conv = [{"role": "user", "content": [{"type": "image", "image": image},
                                              {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        image_inputs = self.processor.process_images(("image", image))
        for key in ("grid_sizes", "merge_sizes"):
            if key in image_inputs and not isinstance(image_inputs[key], torch.Tensor):
                image_inputs[key] = torch.as_tensor(image_inputs[key])
        text_inputs = self.processor.process_text(text, image_inputs, return_tensors="pt")
        inputs = {**text_inputs, **image_inputs}
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        if "pixel_values" in inputs:
            vdtype = next(self.model.get_model().get_vision_encoder().parameters()).dtype
            inputs["pixel_values"] = inputs["pixel_values"].to(vdtype)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False, use_cache=True)
        full = self.processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        answer, conf, reason = parse_response(full)
        return {"answer": answer, "confidence": conf, "reasoning": reason, "raw": full}


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
    gen = HuluMedGenerator(device=args.device)

    for idx, item in enumerate(tqdm(rows, desc=f"Hulu-Med {args.dataset}")):
        if idx < len(results):
            continue
        pred = gen.generate(item["image"], item["question"])
        results.append({"question": item["question"], "gt": item["answer"],
                        "answer": pred["answer"], "confidence": pred["confidence"],
                        "reasoning": pred["reasoning"], "raw": pred["raw"]})
        if len(results) % 25 == 0:
            out.write_text(json.dumps(results, indent=2))
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
