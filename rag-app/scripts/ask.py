"""CLI: python -m scripts.ask "your question"  (run from repo root)"""
import sys
import json
from app.rag import RAG


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "What is this corpus about?"
    out = RAG().answer(question)
    out.pop("hits", None)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
