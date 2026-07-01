"""Evaluation harness — three layers:
  1. Retrieval IR: Hit Rate / Recall@k, MRR, nDCG@k, context precision
  2. Answer quality: faithfulness + relevance via Groq LLM-as-judge; EM/F1 vs gold
  3. Latency: p50/p95 retrieval, end-to-end p95

Run from repo root:  python -m eval.run
Writes eval/results.json
"""
import json
import math
import os
import re
import string
import statistics as st
import time

from groq import Groq
from app.rag import RAG
from app import config

GOLD = os.path.join(os.path.dirname(__file__), "gold.json")
OUT = os.path.join(os.path.dirname(__file__), "results.json")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen/qwen3.6-27b")  # different family than generator (gpt-oss)


# ---------- IR metrics ----------
def dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(hits_sources, relevant, k):
    rels = [1.0 if s == relevant else 0.0 for s in hits_sources[:k]]
    idcg = dcg(sorted(rels, reverse=True)) or 1.0
    return dcg(rels) / idcg


# ---------- text normalization for EM/F1 ----------
def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = {}
    for w in p:
        if w in g:
            common[w] = min(p.count(w), g.count(w))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def em(pred, gold):
    return float(_norm(gold) in _norm(pred))


# ---------- LLM-as-judge for faithfulness + relevance ----------
JUDGE_SYS = (
    "You are a strict evaluator. Given a QUESTION, the CONTEXT passages an answer "
    "was allowed to use, and the ANSWER, score two things from 1 to 5.\n"
    "faithfulness: is every claim in the ANSWER supported by the CONTEXT? "
    "(5 = fully grounded, 1 = fabricated).\n"
    "relevance: does the ANSWER actually address the QUESTION? "
    "(5 = directly answers, 1 = off-topic).\n"
    "Return ONLY compact JSON: {\"faithfulness\": int, \"relevance\": int}"
)


def judge(client, question, context, answer):
    msg = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"
    resp = client.chat.completions.create(
        model=JUDGE_MODEL, temperature=0.0,
        messages=[{"role": "system", "content": JUDGE_SYS},
                  {"role": "user", "content": msg}],
    )
    raw = resp.choices[0].message.content.strip()
    if "</think>" in raw:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    elif "<think>" in raw:
        bt = re.findall(r"`(\{[^`]+\})`", raw, re.S)
        if bt:
            try:
                d = json.loads(bt[-1])
                return int(d["faithfulness"]), int(d["relevance"])
            except Exception:
                pass
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0))
        return int(d["faithfulness"]), int(d["relevance"])
    except Exception:
        return None, None


def main():
    rag = RAG()
    client = Groq(api_key=config.GROQ_API_KEY)
    gold = json.load(open(GOLD))
    K = config.TOP_K

    hit, rr, ndcgs, ctx_prec = [], [], [], []
    faiths, rels, ems, f1s = [], [], [], []
    ret_ms, e2e_ms = [], []
    no_ctx_correct = []
    rows = []

    for item in gold:
        q, relevant, goldans = item["q"], item["relevant"], item["gold"]
        t0 = time.perf_counter()
        out = rag.answer(q, k=K)
        e2e = (time.perf_counter() - t0) * 1000
        ret_ms.append(out["retrieval_ms"])
        e2e_ms.append(e2e)

        hits = out.get("hits", [])
        sources = [h["metadata"].get("source") for h in hits]

        if relevant is None:
            # out-of-corpus probe: success = guard fired
            no_ctx_correct.append(1.0 if out["no_context"] else 0.0)
        else:
            found = relevant in sources
            hit.append(1.0 if found else 0.0)
            rr.append(1.0 / (sources.index(relevant) + 1) if found else 0.0)
            ndcgs.append(ndcg_at_k(sources, relevant, K))
            # context precision: fraction of retrieved chunks from the relevant doc
            if sources:
                ctx_prec.append(sum(1 for s in sources if s == relevant) / len(sources))
            ans = out["answer"]
            ctx = "\n".join(h["text"] for h in hits)
            fa, re_ = judge(client, q, ctx, ans)
            if fa:
                faiths.append(fa)
            if re_:
                rels.append(re_)
            ems.append(em(ans, goldans))
            f1s.append(f1(ans, goldans))

        rows.append({"q": q, "relevant": relevant,
                     "retrieved": sources, "no_context": out["no_context"],
                     "answer": out["answer"]})

    def p(vals, q):
        return round(st.quantiles(vals, n=100)[q - 1], 1) if len(vals) > 1 else round(vals[0], 1)

    results = {
        "config": {"k": K, "embed_model": config.EMBED_MODEL,
                   "embed_dim": config.EMBED_DIM, "generator": config.GROQ_MODEL,
                   "judge": JUDGE_MODEL, "n_questions": len(gold)},
        "retrieval": {
            "hit_rate@k": round(st.mean(hit), 3),
            "mrr": round(st.mean(rr), 3),
            "ndcg@k": round(st.mean(ndcgs), 3),
            "context_precision": round(st.mean(ctx_prec), 3),
        },
        "answer": {
            "faithfulness_mean_5": round(st.mean(faiths), 2) if faiths else None,
            "relevance_mean_5": round(st.mean(rels), 2) if rels else None,
            "exact_match": round(st.mean(ems), 3),
            "f1": round(st.mean(f1s), 3),
        },
        "no_context_guard_accuracy": round(st.mean(no_ctx_correct), 3) if no_ctx_correct else None,
        "latency_ms": {
            "retrieval_p50": p(ret_ms, 50), "retrieval_p95": p(ret_ms, 95),
            "e2e_p95": p(e2e_ms, 95),
        },
        "rows": rows,
    }
    json.dump(results, open(OUT, "w"), indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "rows"}, indent=2))
    print(f"\nFull per-question rows written to {OUT}")


if __name__ == "__main__":
    main()
