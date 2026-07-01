"""Retrieval + grounded generation with citations and a no-context guard."""
import time
from groq import Groq
from . import config
from .store import Store

_SYSTEM = (
    "You are a precise QA assistant. Answer ONLY from the numbered context "
    "passages provided. Cite the passages you use inline as [1], [2], etc. "
    "If the context does not contain the answer, reply exactly: "
    "\"I don't have enough information in the provided context to answer that.\" "
    "Do not use outside knowledge."
)


class RAG:
    def __init__(self):
        self.store = Store()
        self._groq = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None

    def _format_context(self, hits):
        return "\n\n".join(
            f"[{i+1}] (source: {h['metadata'].get('source')}) {h['text']}"
            for i, h in enumerate(hits)
        )

    def answer(self, question: str, k: int | None = None, where: dict | None = None):
        k = k or config.TOP_K
        t0 = time.perf_counter()
        hits = self.store.query(question, k=k, where=where)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        relevant = [h for h in hits if h["score"] >= config.MIN_SCORE]

        # no-context guard: don't even call the LLM if nothing clears the floor
        if not relevant:
            return {
                "answer": "I don't have enough information in the provided "
                          "context to answer that.",
                "citations": [], "chunks_used": 0,
                "retrieval_ms": round(retrieval_ms, 1),
                "generation_ms": 0.0, "tokens": {}, "no_context": True,
                "hits": hits,
            }

        context = self._format_context(relevant)
        prompt = f"Context passages:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        if not self._groq:
            raise RuntimeError("GROQ_API_KEY not set")

        t1 = time.perf_counter()
        resp = self._groq.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0.0,
        )
        generation_ms = (time.perf_counter() - t1) * 1000
        usage = resp.usage

        return {
            "answer": resp.choices[0].message.content.strip(),
            "citations": [
                {"n": i + 1, "source": h["metadata"].get("source"),
                 "chunk_index": h["metadata"].get("chunk_index"),
                 "score": round(h["score"], 3)}
                for i, h in enumerate(relevant)
            ],
            "chunks_used": len(relevant),
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": round(generation_ms, 1),
            "tokens": {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens,
            },
            "no_context": False,
            "hits": relevant,
        }
