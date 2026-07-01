"""Ingestion: PDF/HTML/MD -> clean text -> overlapping char chunks.

Idempotency: each chunk's vector ID is sha256(source_path + chunk_index + text).
Re-ingesting identical content produces identical IDs; Chroma upsert overwrites
rather than duplicating, so vector count stays flat on re-run.
"""
import hashlib
import os
from dataclasses import dataclass

from pypdf import PdfReader
from bs4 import BeautifulSoup
import markdown as md_lib
from . import config


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


# ---------- loaders ----------
def _load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _load_html(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _load_md(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = md_lib.markdown(f.read())
    return BeautifulSoup(html, "html.parser").get_text(separator="\n")


_LOADERS = {".pdf": _load_pdf, ".html": _load_html, ".htm": _load_html, ".md": _load_md}


def load_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext not in _LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return _LOADERS[ext](path)


# ---------- chunking ----------
def _normalize(text: str) -> str:
    # collapse runs of whitespace, keep paragraph breaks
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return text


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = _normalize(text)
    if not text:
        return []
    chunks, start, n = [], 0, len(text)
    step = max(1, size - overlap)
    while start < n:
        chunks.append(text[start:start + size])
        start += step
    return chunks


def _chunk_id(source: str, idx: int, text: str) -> str:
    h = hashlib.sha256(f"{source}::{idx}::{text}".encode("utf-8")).hexdigest()
    return h[:32]


def build_chunks(path: str, size=None, overlap=None) -> list[Chunk]:
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    raw = load_text(path)
    pieces = chunk_text(raw, size, overlap)
    source = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    out = []
    for i, piece in enumerate(pieces):
        out.append(Chunk(
            id=_chunk_id(source, i, piece),
            text=piece,
            metadata={"source": source, "doc_type": ext, "chunk_index": i},
        ))
    return out


def iter_corpus(folder: str):
    for root, _, files in os.walk(folder):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in _LOADERS:
                yield os.path.join(root, fn)
