"""The judge core: prompt construction, structured-verdict parsing with
malformed-JSON recovery, few-shot calibration anchors, and full audit logging.

Two modes implemented:
  - pointwise: score a single output against the rubric
  - pairwise:  compare A vs B (used with position-bias order swap)
"""
import json
import os
import re
import time
import uuid

from groq import Groq, RateLimitError
from . import config
from .rubric import RUBRIC, rubric_text, weighted_overall

_client = None


def client():
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


# Few-shot anchors fight score clustering: show the judge what a 2 and a 5 look like.
_FEWSHOT = (
    "Calibration examples (so you use the full 1-5 range):\n"
    "EX1 -> answer is fluent but factually wrong: "
    '{"correctness":1,"faithfulness":2,"completeness":3,'
    '"instruction_following":4,"tone_safety":5}\n'
    "EX2 -> answer is correct, grounded, complete: "
    '{"correctness":5,"faithfulness":5,"completeness":5,'
    '"instruction_following":5,"tone_safety":5}\n'
)

_POINTWISE_SYS = (
    "You are a rigorous evaluation judge. Score the ANSWER against each rubric "
    "criterion from 1 to 5 using the anchors. Judge substance, not length or "
    "confidence: a longer or more confident answer is NOT automatically better, "
    "and an unsupported claim must lower faithfulness even if it sounds right.\n\n"
    f"RUBRIC:\n{rubric_text()}\n\n{_FEWSHOT}\n"
    "Return ONLY valid JSON, no prose, with this exact shape:\n"
    '{"scores": {"correctness": int, "faithfulness": int, "completeness": int, '
    '"instruction_following": int, "tone_safety": int}, '
    '"rationale": {"correctness": str, "faithfulness": str, "completeness": str, '
    '"instruction_following": str, "tone_safety": str}}'
)

_PAIRWISE_SYS = (
    "You are a rigorous pairwise judge. You will see a QUESTION and two answers, "
    "ANSWER_A and ANSWER_B. Decide which better satisfies the rubric. Do NOT favor "
    "the longer, more confident, or first-listed answer; judge substance and "
    "grounding.\n\n"
    f"RUBRIC:\n{rubric_text()}\n\n"
    'Return ONLY JSON: {"winner": "A" | "B" | "tie", "rationale": str}'
)


def _extract_json(raw: str):
    """Robust recovery from malformed JSON: strip fences, grab outermost braces.

    Handles three Qwen 3 output shapes:
      1. <think>...</think> followed by actual JSON  (clean case)
      2. <think>... with no closing tag — JSON is in backticks inside the block
      3. Plain JSON / ```json fenced JSON
    """
    if raw is None:
        return None
    s = raw.strip()

    if "</think>" in s:
        # Strip the thinking block; real answer follows the closing tag
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()
    elif "<think>" in s:
        # No closing tag — JSON was written in backticks inside the think block.
        # Extract the last backtick-fenced {...} which is the final answer.
        bt_matches = re.findall(r"`(\{[^`]+\})`", s, re.S)
        if bt_matches:
            candidate = bt_matches[-1]
            try:
                return json.loads(candidate)
            except Exception:
                candidate2 = re.sub(r",\s*([}\]])", r"\1", candidate)
                try:
                    return json.loads(candidate2)
                except Exception:
                    pass
        # Fall through to greedy search below

    s = re.sub(r"^```(json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    # grab the outermost {...}
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    frag = m.group(0)
    try:
        return json.loads(frag)
    except Exception:
        # last resort: trailing-comma cleanup
        frag2 = re.sub(r",\s*([}\]])", r"\1", frag)
        try:
            return json.loads(frag2)
        except Exception:
            return None


def _log(record: dict):
    os.makedirs(config.LOG_DIR, exist_ok=True)
    path = os.path.join(config.LOG_DIR, f"{record['id']}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def _call(system, user):
    t0 = time.perf_counter()
    for attempt in range(5):
        try:
            resp = client().chat.completions.create(
                model=config.JUDGE_MODEL, temperature=config.JUDGE_TEMPERATURE,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            ms = (time.perf_counter() - t0) * 1000
            return resp.choices[0].message.content, resp.usage, ms
        except RateLimitError:
            wait = 15 * (attempt + 1)
            print(f"  [rate limit] waiting {wait}s …")
            time.sleep(wait)
    raise RuntimeError("Rate limit retries exhausted")


def judge_pointwise(case: dict) -> dict:
    """case: {input, system_prompt, model_output, expected_output?, criteria?}"""
    user = (
        f"QUESTION:\n{case['input']}\n\n"
        f"SYSTEM PROMPT GIVEN TO THE MODEL:\n{case.get('system_prompt','(none)')}\n\n"
        f"CONTEXT/EXPECTED (if any):\n{case.get('expected_output','(none)')}\n\n"
        f"ANSWER TO JUDGE:\n{case['model_output']}"
    )
    rid = uuid.uuid4().hex[:12]
    parsed, raw, usage, ms = None, None, None, None
    for attempt in range(config.MAX_RETRIES + 1):
        raw, usage, ms = _call(_POINTWISE_SYS, user)
        parsed = _extract_json(raw)
        if parsed and "scores" in parsed:
            break
    record = {"id": rid, "mode": "pointwise", "case_input": case["input"],
              "judge_system_prompt": _POINTWISE_SYS, "judge_user_prompt": user,
              "judge_model": config.JUDGE_MODEL, "raw_response": raw,
              "parsed_ok": bool(parsed and "scores" in parsed),
              "tokens": usage.total_tokens if usage else None, "latency_ms": round(ms, 1)}
    _log(record)

    if not parsed or "scores" not in parsed:
        return {"ok": False, "id": rid, "raw": raw,
                "overall": None, "scores": {}, "rationale": {}}

    scores = {k: int(v) for k, v in parsed["scores"].items() if k in RUBRIC}
    return {"ok": True, "id": rid, "scores": scores,
            "rationale": parsed.get("rationale", {}),
            "overall": weighted_overall(scores),
            "tokens": usage.total_tokens, "latency_ms": round(ms, 1)}


def judge_pairwise(question, answer_a, answer_b, system_prompt="(none)") -> dict:
    user = (f"QUESTION:\n{question}\n\nSYSTEM PROMPT:\n{system_prompt}\n\n"
            f"ANSWER_A:\n{answer_a}\n\nANSWER_B:\n{answer_b}")
    rid = uuid.uuid4().hex[:12]
    raw, usage, ms = _call(_PAIRWISE_SYS, user)
    parsed = _extract_json(raw) or {}
    winner = parsed.get("winner", "tie")
    if winner not in ("A", "B", "tie"):
        winner = "tie"
    _log({"id": rid, "mode": "pairwise", "question": question,
          "judge_model": config.JUDGE_MODEL, "raw_response": raw,
          "winner": winner, "tokens": usage.total_tokens if usage else None,
          "latency_ms": round(ms, 1)})
    return {"id": rid, "winner": winner, "rationale": parsed.get("rationale", ""),
            "tokens": usage.total_tokens, "latency_ms": round(ms, 1)}
