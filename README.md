# Applied AI / ML Engineering — Take-Home Assignment

**Candidate:** Ashutosh Yadav · yadavashutosh162@gmail.com

Two end-to-end systems built and evaluated from scratch:

| | Problem | Folder |
|---|---|---|
| 1 | Cost-Efficient RAG Application | [`rag-app/`](rag-app/) |
| 2 | LLM-as-Judge Evaluation Pipeline | [`llm-judge/`](llm-judge/) |

---

## Problem 1 — Cost-Efficient RAG Application

A QA service over a Kubernetes documentation corpus backed by **ChromaDB** (embedded, zero standing pods).

**Key results:**

| Metric | Value |
|---|---|
| Hit Rate@5 | **1.000** |
| MRR | **1.000** |
| nDCG@5 | **0.991** |
| Faithfulness (LLM-judge, 1–5) | **5.0 / 5** |
| Answer Relevance (LLM-judge, 1–5) | **5.0 / 5** |
| No-context guard accuracy | **1.000** |
| Cost vs managed DB at 100K vectors | **83% cheaper** |
| Cost vs managed DB at 10M vectors | managed DB wins (crossover shown) |

**Stack:** ChromaDB · all-MiniLM-L6-v2 (384-dim ONNX) · Groq `openai/gpt-oss-120b` · FastAPI

**Quick start:**
```bash
cd rag-app
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here

python -m scripts.ingest ./data          # ingest corpus
python -m eval.retrieval_only            # retrieval metrics (no key needed)
python -m eval.run                       # full 3-layer eval
python -m scripts.cost                   # cost comparison table
uvicorn app.server:app --port 8000       # HTTP endpoint
```

→ Full details, architecture diagram, and discussion in [`rag-app/README.md`](rag-app/README.md)

---

## Problem 2 — LLM-as-Judge Evaluation Pipeline

A judging pipeline that scores `{ input, system_prompt, model_output, expected_output }` with a structured per-criterion verdict — and takes the judge's own biases seriously.

**Key results:**

| Check | Result |
|---|---|
| Agreement with gold labels | **100%** |
| Cohen's kappa | **1.0** |
| Position flip rate (A/B order bias) | **0.0%** |
| Adversarial fooled rate | **0** (verbose-but-wrong and confidently-wrong both caught) |
| Verbosity score inflation | **−0.3** (padding lowered the score) |
| Test-retest flip rate (temp=0) | **0.0%** |
| A/B winner | `prompt_v1_concise` (win rate **0.8**) |

**Bias mitigations implemented:** position order-swap · verbosity probe · cross-family judge (Qwen vs GPT-OSS) · few-shot calibration anchors · per-criterion grounding requirement

**Stack:** Groq `qwen/qwen3.6-27b` (judge) · `openai/gpt-oss-120b` (generator) · pointwise + pairwise modes

**Quick start:**
```bash
cd llm-judge
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here

python -m scripts.run_suite suites/k8s_suite.json   # suite → report
python -m scripts.run_ab    suites/ab_pairs.json    # A/B comparison
python -m scripts.validate_judge                    # agreement + adversarial
```

→ Full details, architecture diagram, bias table, and discussion in [`llm-judge/README.md`](llm-judge/README.md)

---

## Repository structure

```
Gen-AI-Assignment/
├── rag-app/
│   ├── app/               # FastAPI server, RAG logic, ChromaDB store
│   ├── data/              # Kubernetes MD corpus (5 docs)
│   ├── eval/              # gold.json, run.py, retrieval_only.py, results*.json
│   ├── scripts/           # ingest.py, ask.py, cost.py
│   ├── docs/              # architecture diagram
│   ├── .env.example       # env var names only — no secrets
│   ├── requirements.txt
│   └── README.md
└── llm-judge/
    ├── judge/             # core.py, rubric.py, bias.py, config.py
    ├── scripts/           # run_suite.py, run_ab.py, validate_judge.py
    ├── suites/            # k8s_suite.json, ab_pairs.json
    ├── reports/           # suite_report.json, ab_report.json, judge_validation.json
    ├── docs/              # architecture diagram
    ├── .env.example       # env var names only — no secrets
    ├── requirements.txt
    └── README.md
```

## Environment variables

Neither `.env` file is committed. Copy the relevant `.env.example`, rename to `.env`, and fill in your key:

```
GROQ_API_KEY=your_key_here   # free tier at console.groq.com
```
