"""Central config. Everything overridable via env; no secrets in code."""
import os

# --- Vector store ---
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION = os.getenv("RAG_COLLECTION", "docs")

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))        # chars
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))  # chars

# --- Embeddings ---
# EMBED_BACKEND: "onnx" (default, Chroma's MiniLM-L6-v2) or "local" (offline fallback)
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "onnx")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = 384

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25"))  # cosine-sim floor for "no context"

# --- LLM (Groq) ---
# Note: llama-3.3-70b-versatile was deprecated by Groq on 2026-06-17.
# Default generator is now gpt-oss-120b. Override with GROQ_MODEL if you have access.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# --- Server ---
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
