# llm-fit-check

One line: which open-source LLM works best on your local machine today.

```bash
pip install llm-fit-check
llm-fit-check --task code
# > For code, run Qwen2.5-Coder 32B (`ollama run qwen2.5-coder:32b`) — ~21.0GB, score 90/100 on the 2026-01-15 leaderboard snapshot.
```

## What it does

1. Detects your hardware (RAM, VRAM, Apple Silicon unified memory, NVIDIA via `pynvml`/`nvidia-smi`).
2. Pulls a daily snapshot of leaderboard scores (Open LLM Leaderboard v2, LiveBench, Aider, EvalPlus) — falls back to a bundled snapshot offline.
3. Filters the model catalog to those that fit your usable memory (Q4_K_M GGUF footprint + headroom).
4. Returns the highest-scoring model for your task in one line.

## Usage

```bash
llm-fit-check                          # general-purpose pick
llm-fit-check --task reasoning --why   # show hardware + runners-up + sources
llm-fit-check --ask                    # interactive: asks what you want to do
llm-fit-check list --task code         # full ranked catalog
llm-fit-check hardware                 # just print detected hardware
llm-fit-check --refresh                # force-refresh leaderboard cache
```

Tasks: `code`, `chat`, `reasoning`, `general`.

## As a library

```python
from llm_fit_check import recommend

rec = recommend(task="code")
print(rec.one_liner())
print(rec.pick.ollama)  # e.g. "qwen2.5-coder:32b"
```

## How the snapshot is refreshed

A GitHub Action (`.github/workflows/refresh.yml`) runs **weekly** (Mon 06:00 UTC)
and regenerates `src/llm_fit_check/data/snapshot.json` by running
`scripts/refresh_snapshot.py`. The package fetches the raw JSON from GitHub on
first use and caches it locally for 7 days. `--offline` uses the bundled
snapshot only.

**Sources** today:
- HuggingFace Open LLM Leaderboard v2 (paginated rows API, with retries
  and early-stop once all known aliases are matched)
- LiveBench `model_judgment` dataset (best-effort; falls back gracefully
  when the rows endpoint 500s — a parquet-based fetch is the next step)

**Adding a new model**: edit `scripts/aliases.yaml` (HF repo id → catalog id)
and `scripts/footprints.yaml` (display name, params, Q4 footprint, ollama tag),
then trigger the workflow manually. Run locally with:

```bash
pip install httpx pyyaml
python scripts/refresh_snapshot.py            # writes snapshot.json
python scripts/refresh_snapshot.py --dry-run  # prints to stdout instead
```

If every source fails, the existing snapshot is preserved (no partial writes).

## Roadmap

- [ ] Live scrapers for LMArena, LiveBench, Aider (currently a curated snapshot)
- [ ] AMD ROCm + Intel Arc detection
- [ ] Quantization-aware footprints (Q8, Q5, Q3)
- [ ] Detect installed runtimes (Ollama, llama.cpp, LM Studio) and recommend models you already have
- [ ] Per-task benchmarks beyond aggregate scores

## License

MIT.
