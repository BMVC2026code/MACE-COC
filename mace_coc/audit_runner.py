"""MACE-CoC audit pipeline: runs 4 agents + meta-adjudicator on generator outputs."""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from mace_coc.prompts import render
from mace_coc.utils import parse_json_object
from mace_coc.datasets import load_dataset_by_name


QWEN_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
LLAMA_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


class QwenAgent:
    """Qwen2.5-VL audit agent (processes image + text)."""

    def __init__(self, device="cuda:0", model_id=QWEN_ID):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=torch.float16, device_map={"": device},
        ).eval()

    def generate(self, image, prompt, max_new_tokens=448):
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False, use_cache=True)
        return self.processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()


class LlamaMetaAgent:
    """Llama-3 text-only meta-adjudicator."""

    def __init__(self, device="cuda:1", model_id=LLAMA_ID, load_in_4bit=False):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = {"trust_remote_code": True, "device_map": {"": device}}
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        else:
            model_kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs).eval()

    def generate(self, prompt, max_new_tokens=256):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False, use_cache=True)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()


def audit_case(agent, meta_agent, image, question, gen):
    """Run 4 audit probes + meta-adjudicator on a single case."""
    blind_raw = agent.generate(image, render("blind_topk", question=question))
    blind = parse_json_object(blind_raw)

    falsification_raw = agent.generate(image, render(
        "falsification", question=question,
        initial_answer=gen["answer"], initial_reasoning=gen["reasoning"]))
    falsification = parse_json_object(falsification_raw)

    alternative_raw = agent.generate(image, render(
        "alternative", question=question,
        initial_answer=gen["answer"],
        blind_evidence=json.dumps(blind or {"raw": blind_raw})))
    alternative = parse_json_object(alternative_raw)

    grounding_raw = agent.generate(image, render(
        "grounding", question=question,
        initial_answer=gen["answer"], initial_reasoning=gen["reasoning"]))
    grounding = parse_json_object(grounding_raw)

    meta_raw = meta_agent.generate(render(
        "meta_calibrator", question=question,
        initial_answer=gen["answer"], initial_confidence=gen["confidence"],
        blind_evidence=json.dumps(blind or {"raw": blind_raw}),
        falsification=json.dumps(falsification or {"raw": falsification_raw}),
        alternative=json.dumps(alternative or {"raw": alternative_raw}),
        grounding=json.dumps(grounding or {"raw": grounding_raw})))
    meta = parse_json_object(meta_raw)

    return {
        "blind_raw": blind_raw, "blind": blind,
        "falsification_raw": falsification_raw, "falsification": falsification,
        "alternative_raw": alternative_raw, "alternative": alternative,
        "grounding_raw": grounding_raw, "grounding": grounding,
        "meta_raw": meta_raw, "meta": meta,
    }


def load_generator_outputs(path, answer_key, confidence_key, reasoning_key):
    raw = Path(path).read_text()
    start = raw.find("[")
    if start > 0:
        raw = raw[start:]
    rows = json.loads(raw)
    return {idx: {"answer": r.get(answer_key, ""),
                  "confidence": r.get(confidence_key, 0.5),
                  "reasoning": r.get(reasoning_key, ""),
                  "question": r.get("question", ""),
                  "gt": r.get("gt", "")}
            for idx, r in enumerate(rows)}


def completed_ids(path):
    p = Path(path)
    if not p.exists():
        return set()
    done = set()
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                done.add(json.loads(line)["case_id"])
            except Exception:
                pass
    return done


def main():
    parser = argparse.ArgumentParser(description="Run MACE-CoC audit pipeline")
    parser.add_argument("--dataset", choices=["vqarad", "slake", "mimic_cxr_vqa_1k"], required=True)
    parser.add_argument("--generator-json", required=True, help="Path to generator output JSON")
    parser.add_argument("--answer-key", required=True)
    parser.add_argument("--confidence-key", required=True)
    parser.add_argument("--reasoning-key", required=True)
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--audit-device", default="cuda:0", help="GPU for Qwen audit agent")
    parser.add_argument("--meta-device", default="cuda:1", help="GPU for meta-adjudicator")
    parser.add_argument("--meta-model", default=LLAMA_ID)
    parser.add_argument("--meta-4bit", action="store_true")
    parser.add_argument("--subset-json", default=None, help="For MIMIC-CXR-VQA: path to subset JSON")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    kwargs = {"split": "test"}
    if args.dataset == "mimic_cxr_vqa_1k":
        kwargs["subset_json"] = args.subset_json
    rows = load_dataset_by_name(args.dataset, **kwargs)
    if args.limit:
        rows = rows[:args.limit]

    gen_outputs = load_generator_outputs(
        args.generator_json, args.answer_key, args.confidence_key, args.reasoning_key)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(args.output)

    agent = QwenAgent(device=args.audit_device)
    meta_agent = LlamaMetaAgent(device=args.meta_device, model_id=args.meta_model,
                                load_in_4bit=args.meta_4bit)

    with open(args.output, "a") as f:
        for idx, item in enumerate(tqdm(rows, desc=f"MACE-CoC {args.dataset}")):
            if item["case_id"] in done:
                continue
            gen = gen_outputs.get(idx)
            if gen is None:
                continue
            audits = audit_case(agent, meta_agent, item["image"], item["question"], gen)
            record = {
                "dataset": item["dataset"], "case_id": item["case_id"],
                "question": item["question"], "gt": item["answer"],
                "generator": gen, **audits,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
