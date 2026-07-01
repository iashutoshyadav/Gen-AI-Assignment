"""CLI: python -m scripts.ingest <folder>  (run from repo root)"""
import sys
import time
from app.store import Store
from app import ingest, config


def main(folder):
    store = Store()
    before = store.count()
    t0 = time.perf_counter()
    files, total = 0, 0
    for path in ingest.iter_corpus(folder):
        chunks = ingest.build_chunks(path)
        store.add_chunks(chunks)
        files += 1
        total += len(chunks)
        print(f"  + {path}: {len(chunks)} chunks")
    after = store.count()
    print(f"\nFiles: {files} | chunks built: {total}")
    print(f"Vectors before: {before} -> after: {after} (delta {after - before})")
    print(f"chunk_size={config.CHUNK_SIZE} overlap={config.CHUNK_OVERLAP} "
          f"took {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "./data")
