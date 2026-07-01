"""Offline deterministic embedding fallback.

The committed default is Chroma's ONNX all-MiniLM-L6-v2 (384-dim). In sandboxed
/ air-gapped environments where the model can't be downloaded, set
EMBED_BACKEND=local to use this hashing-based bag-of-words embedder. It is NOT as
good as MiniLM semantically, but it is fully offline, deterministic, and lets the
pipeline + eval harness run end-to-end anywhere. Same 384 dims for drop-in parity.
"""
import hashlib
import math
import re
from collections import Counter

DIM = 384
_token_re = re.compile(r"[a-z0-9]+")


def _tokens(text: str):
    return _token_re.findall(text.lower())


class LocalHashingEmbedding:
    """Chroma-compatible embedding function (callable taking list[str])."""

    def name(self) -> str:                      # Chroma 1.x requires this
        return "local-hashing-384"

    def __call__(self, input):
        return [self._embed(t) for t in input]

    # Chroma 1.5+ split interface
    def embed_documents(self, input):
        return [self._embed(t) for t in input]

    def embed_query(self, input):
        if isinstance(input, str):
            return self._embed(input)
        return [self._embed(t) for t in input]

    def _embed(self, text: str):
        vec = [0.0] * DIM
        counts = Counter(_tokens(text))
        for tok, c in counts.items():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % DIM
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(c))
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
