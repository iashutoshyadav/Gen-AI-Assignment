"""Thin wrapper over a persistent ChromaDB collection.

Uses Chroma's default embedding function (ONNX all-MiniLM-L6-v2, 384-dim),
so no torch / no per-token embedding cost. Cosine space.
"""
import chromadb
from chromadb.utils import embedding_functions
from . import config, ingest


class Store:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        if config.EMBED_BACKEND == "local":
            from .local_embed import LocalHashingEmbedding
            self.ef = LocalHashingEmbedding()
        else:
            self.ef = embedding_functions.DefaultEmbeddingFunction()  # MiniLM-L6-v2 ONNX
        self.col = self.client.get_or_create_collection(
            name=config.COLLECTION,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.col.count()

    def add_chunks(self, chunks: list[ingest.Chunk]):
        if not chunks:
            return 0
        # upsert => idempotent on identical IDs
        self.col.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        return len(chunks)

    def query(self, text: str, k: int, where: dict | None = None):
        res = self.col.query(
            query_texts=[text],
            n_results=k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        if res["ids"] and res["ids"][0]:
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0],
                res["metadatas"][0], res["distances"][0]
            ):
                # cosine distance -> similarity
                hits.append({
                    "id": cid, "text": doc, "metadata": meta,
                    "score": 1.0 - dist,
                })
        return hits
