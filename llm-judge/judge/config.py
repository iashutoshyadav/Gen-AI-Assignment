"""Config for the LLM-as-judge pipeline. Env-driven, no secrets in code."""
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Judge and generator are configured INDEPENDENTLY (self-enhancement mitigation).
# They are deliberately from DIFFERENT model families: a judge from the same
# family as the generator tends to over-reward its own outputs.
# Note: llama-3.3-70b-versatile was deprecated by Groq 2026-06-17.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen/qwen3.6-27b")            # judge family: Qwen
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "openai/gpt-oss-120b")  # generator family: GPT-OSS

JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
MAX_RETRIES = int(os.getenv("JUDGE_MAX_RETRIES", "2"))

LOG_DIR = os.getenv("JUDGE_LOG_DIR", "./reports/logs")
