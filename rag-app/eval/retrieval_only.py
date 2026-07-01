"""Retrieval-only eval (no LLM needed). Runs fully offline.
   python -m eval.retrieval_only
Produces the IR-metrics layer so retrieval can be validated without an API key.
"""
import json
import math
import os
import statistics as st
import time
from app.store import Store
from app import config

GOLD = os.path.join(os.path.dirname(__file__), "gold.json")


def dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(sources, relevant, k):
    rels = [1.0 if s == relevant else 0.0 for s in sources[:k]]
    idcg = dcg(sorted(rels, reverse=True)) or 1.0
    return dcg(rels) / idcg


def main():
    store = Store()
    gold = json.load(open(GOLD))
    K = config.TOP_K
    hit, rr, ndcgs, ctx_prec, ret_ms, guard = [], [], [], [], [], []

    for item in gold:
        q, relevant = item["q"], item["relevant"]
        t0 = time.perf_counter()
        hits = store.query(q, k=K)
        ret_ms.append((time.perf_counter() - t0) * 1000)
        relevant_hits = [h for h in hits if h["score"] >= config.MIN_SCORE]
        sources = [h["metadata"].get("source") for h in hits]

        if relevant is None:
            guard.append(1.0 if not relevant_hits else 0.0)
            continue
        found = relevant in sources
        hit.append(1.0 if found else 0.0)
        rr.append(1.0 / (sources.index(relevant) + 1) if found else 0.0)
        ndcgs.append(ndcg_at_k(sources, relevant, K))
        ctx_prec.append(sum(1 for s in sources if s == relevant) / len(sources) if sources else 0)

    def p(v, q):
        return round(st.quantiles(v, n=100)[q - 1], 2) if len(v) > 1 else round(v[0], 2)

    out = {
        "backend": config.EMBED_BACKEND, "k": K,
        "hit_rate@k": round(st.mean(hit), 3),
        "mrr": round(st.mean(rr), 3),
        "ndcg@k": round(st.mean(ndcgs), 3),
        "context_precision": round(st.mean(ctx_prec), 3),
        "no_context_guard_accuracy": round(st.mean(guard), 3) if guard else None,
        "retrieval_p50_ms": p(ret_ms, 50), "retrieval_p95_ms": p(ret_ms, 95),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
