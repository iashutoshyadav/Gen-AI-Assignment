"""HTTP endpoint. Logs per-query latency, chunk count, token usage."""
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from .rag import RAG
from . import config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag")

app = FastAPI(title="Cost-Efficient RAG")
_rag = None


def get_rag():
    global _rag
    if _rag is None:
        _rag = RAG()
    return _rag


class Query(BaseModel):
    question: str
    k: int | None = None
    doc_type: str | None = None  # metadata filter, e.g. "pdf"


@app.get("/health")
def health():
    return {"status": "ok", "vectors": get_rag().store.count(),
            "model": config.GROQ_MODEL, "embed_dim": config.EMBED_DIM}


@app.post("/query")
def query(q: Query):
    where = {"doc_type": q.doc_type} if q.doc_type else None
    out = get_rag().answer(q.question, k=q.k, where=where)
    log.info("q=%r chunks=%d ret_ms=%.1f gen_ms=%.1f tokens=%s no_ctx=%s",
             q.question[:60], out["chunks_used"], out["retrieval_ms"],
             out["generation_ms"], out["tokens"], out["no_context"])
    out.pop("hits", None)
    return out
