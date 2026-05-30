JSON_ONLY = "Return valid JSON only. No markdown. No extra text."


BLIND_TOPK_PROMPT = """You are a board-certified radiologist.

Question: {question}

Review the image and question without seeing the AI answer.
Use only visible medical image evidence.
Give up to 3 plausible short answers.
Do not use generic clinical priors unless the image supports them.

{json_only}

Schema:
{{
  "top_answers": [
    {{"answer": "...", "visual_evidence": "...", "support": "weak|moderate|strong"}}
  ],
  "image_sufficiency": "insufficient|partial|sufficient"
}}"""


FALSIFICATION_PROMPT = """You are a senior radiologist asked to find reasons an AI answer may be wrong.

Question: {question}
Initial answer: {initial_answer}
Initial reasoning: {initial_reasoning}

Find medical image evidence that could make the initial answer wrong.
Do not defend the answer.
Focus on visual contradiction, missing required findings, and unsupported radiology claims.

{json_only}

Schema:
{{
  "contradiction": "none|weak|moderate|strong",
  "contradicting_evidence": "...",
  "missing_evidence": "...",
  "unsupported_claims": ["..."]
}}"""


ALTERNATIVE_PROMPT = """You are a senior radiologist considering the strongest alternative answer supported by the image.

Question: {question}
Initial answer: {initial_answer}
Blind evidence:
{blind_evidence}

Find the strongest medically plausible alternative answer and compare it with the initial answer using image evidence.

{json_only}

Schema:
{{
  "alternative_answer": "...",
  "alternative_support": "none|weak|moderate|strong",
  "initial_vs_alternative": "initial_stronger|similar|alternative_stronger|unclear",
  "reason": "..."
}}"""


GROUNDING_PROMPT = """You are a senior radiologist checking whether an AI explanation is grounded in visible image findings.

Question: {question}
Initial answer: {initial_answer}
Initial reasoning: {initial_reasoning}

Does the reasoning depend on visible medical image evidence?
Separate image-grounded claims from generic medical prior claims.

{json_only}

Schema:
{{
  "grounding": "absent|weak|moderate|strong",
  "image_specific_evidence": "...",
  "generic_prior_claims": ["..."]
}}"""

META_CALIBRATOR_PROMPT = """You are a medical VQA confidence adjudicator.

Your task is to estimate the probability that the fixed answer is correct.
You must not change, rewrite, or replace the fixed answer.
You must not introduce new image findings beyond the audit reports.
Use only the question, fixed answer, initial confidence, and structured audit reports.

The initial confidence is only a prior signal. It is not reliable by itself.
The final confidence should reflect whether the fixed answer is visually supported
according to the audit evidence.

Decision principles:
- High confidence requires clear visual support, weak or no contradiction,
  weak alternative-answer pressure, and image-grounded reasoning.
- Low confidence is appropriate when the audits report visually decisive
  contradiction, strong alternative support, poor grounding, or insufficient evidence.
- A reported contradiction is not automatically decisive. Decide whether it is
  visually decisive, plausible but uncertain, or weak/speculative.
- Do not make a binary trust decision. Estimate a calibrated probability.
- Do not average the audit outputs mechanically. Integrate them according to
  their severity and consistency.
- Use 0.0 only when the fixed answer is essentially impossible from the audit evidence.
- Use 1.0 only when the fixed answer is essentially certain from the audit evidence.
- If evidence is mixed, conflicting, incomplete, or internally inconsistent,
  choose a moderate confidence rather than an extreme value.

Audit consistency rule:
- First check whether audit labels and explanations agree.
- If audit labels and explanations conflict, treat the audit packet as uncertain.
- Use extreme confidence only when multiple audit signals agree and their
  explanations are consistent.
- If the audit packet is mixed or internally inconsistent, reduce certainty.

Question: {question}

Fixed answer: {initial_answer}

Initial confidence: {initial_confidence}

Blind evidence:
{blind_evidence}

Falsification:
{falsification}

Alternative:
{alternative}

Grounding:
{grounding}

{json_only}

Schema:
{{
  "evidence_alignment": "low|moderate|high",
  "risk_summary": "...",
  "trust_summary": "...",
  "audit_consistency": "consistent|mixed|internally_inconsistent",
  "contradiction_decisiveness": "none|weak_speculative|plausible_uncertain|visually_decisive",
  "dominant_risk": "none|insufficient_evidence|contradiction|alternative|poor_grounding",
  "final_confidence": 0.0
}}

The field final_confidence must be a single numeric value between 0 and 1.
Return valid JSON only."""


PROMPTS = {
    "blind_topk": BLIND_TOPK_PROMPT,
    "falsification": FALSIFICATION_PROMPT,
    "alternative": ALTERNATIVE_PROMPT,
    "grounding": GROUNDING_PROMPT,
    "meta_calibrator": META_CALIBRATOR_PROMPT,
}


def render(name, **kwargs):
    return PROMPTS[name].format(json_only=JSON_ONLY, **kwargs)
