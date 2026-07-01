"""Reproducible cost model: ChromaDB (self-hosted on one VM) vs a managed,
always-on vector DB, across 100K / 1M / 10M vectors.

ALL ASSUMPTIONS ARE STATED HERE. Numbers are list-price estimates (2026,
us-east); treat as order-of-magnitude, not quotes. Run: python -m scripts.cost
"""

# ---- shared assumptions ----
EMBED_DIM = 384
BYTES_PER_FLOAT = 4
# raw vector bytes + ~30% overhead for HNSW graph + metadata/payload
OVERHEAD = 1.30


def vector_gb(n):
    raw = n * EMBED_DIM * BYTES_PER_FLOAT * OVERHEAD
    return raw / (1024 ** 3)


# ---- ChromaDB self-hosted: cost = the VM it shares with your app ----
# Assumption: Chroma runs embedded / as a sidecar on a VM you already pay for.
# We attribute only the *incremental* RAM/disk it needs. Pricing: generic cloud
# VM at ~$0.0084/GB-RAM-hr equiv; we bill a whole right-sized instance.
def chroma_monthly(n):
    gb = vector_gb(n)
    # pick the smallest instance whose RAM comfortably holds the index (2.5x headroom)
    needed_ram = gb * 2.5
    tiers = [(2, 12), (4, 24), (8, 48), (16, 95), (32, 190), (64, 380)]  # (GB_RAM, $/mo)
    for ram, price in tiers:
        if ram >= max(1.0, needed_ram):
            return price, ram, gb
    return 760, 128, gb  # fallback large box


# ---- Managed vector DB: always-on pods billed by stored vectors ----
# Assumption: a managed serverless/pod plan. Common shape: ~$0.025/GB-mo storage
# PLUS an always-on minimum pod/read-unit charge that dominates at low scale.
MANAGED_MIN_POD = 70.0      # smallest always-on pod / mo
MANAGED_PER_GB = 0.33       # $/GB-mo of indexed vectors (incl. replication x2)


def managed_monthly(n):
    gb = vector_gb(n)
    return max(MANAGED_MIN_POD, MANAGED_MIN_POD + gb * MANAGED_PER_GB * 2), gb


if __name__ == "__main__":
    print(f"{'Vectors':>10} | {'idx GB':>7} | {'Chroma $/mo':>12} | "
          f"{'Managed $/mo':>12} | {'Savings':>8}")
    print("-" * 65)
    for n in (100_000, 1_000_000, 10_000_000):
        c, ram, gb = chroma_monthly(n)
        m, _ = managed_monthly(n)
        save = f"{(1 - c / m) * 100:.0f}%"
        print(f"{n:>10,} | {gb:>7.2f} | {c:>10}({ram}GB) | {m:>12.0f} | {save:>8}")
    print("\nAssumptions: 384-dim float32 vectors, +30% index/metadata overhead,")
    print("Chroma billed as a right-sized shared VM, managed DB = always-on pod")
    print(f"(${MANAGED_MIN_POD}/mo min) + ${MANAGED_PER_GB}/GB-mo x2 replication.")
