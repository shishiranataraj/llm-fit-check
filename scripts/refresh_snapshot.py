"""Regenerate src/llm_fit_check/data/snapshot.json from upstream leaderboards.

Sources:
  - HuggingFace Open LLM Leaderboard v2 (datasets API, no auth required)
  - LiveBench (HF dataset)

The script is intentionally defensive:
  - Each source is fetched in isolation; one failing does not abort the run.
  - If *all* sources fail, the existing snapshot is preserved (exit 1).
  - Models emitted by upstream that aren't in scripts/aliases.yaml are logged
    and skipped — adding a new model = one entry in aliases.yaml + footprints.yaml.

Run:
    python scripts/refresh_snapshot.py
    python scripts/refresh_snapshot.py --dry-run    # print, don't write
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = ROOT / "src" / "llm_fit_check" / "data" / "snapshot.json"
ALIASES_PATH = ROOT / "scripts" / "aliases.yaml"
FOOTPRINTS_PATH = ROOT / "scripts" / "footprints.yaml"

# Map upstream benchmarks -> our task categories. A benchmark can feed >1 task.
BENCHMARK_TASKS: dict[str, tuple[str, ...]] = {
    # Open LLM Leaderboard v2
    "MMLU-PRO": ("general",),
    "GPQA":     ("reasoning", "general"),
    "MATH":     ("reasoning",),
    "MUSR":     ("reasoning",),
    "BBH":      ("reasoning", "general"),
    "IFEval":   ("chat", "general"),
    # LiveBench categories
    "coding":     ("code",),
    "reasoning":  ("reasoning",),
    "language":   ("chat",),
    "instruction_following": ("chat",),
    "data_analysis": ("general",),
    "mathematics": ("reasoning",),
}

log = logging.getLogger("refresh")


@dataclass
class RawScores:
    """Per-model raw benchmark scores keyed by (catalog_id, benchmark) -> 0..100."""
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, catalog_id: str, benchmark: str, score: float) -> None:
        self.by_model.setdefault(catalog_id, {})[benchmark] = score


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_alias_index(aliases_yaml: dict) -> dict[str, str]:
    return {k.lower(): v for k, v in aliases_yaml["aliases"].items()}


# ---------- Source: HF Open LLM Leaderboard v2 ----------

OPEN_LLM_PAGE_SIZE = 100
OPEN_LLM_MAX_PAGES = 200  # ~20k rows, plenty to cover every official model
OPEN_LLM_RETRIES_PER_PAGE = 3


def fetch_open_llm_leaderboard(client: httpx.Client, alias_idx: dict[str, str]) -> RawScores:
    """Paginate the datasets-server rows API until we've matched every alias
    or exhausted the leaderboard (the leaderboard contains thousands of
    community fine-tunes; official models are scattered alphabetically)."""
    out = RawScores()
    targets_remaining = set(alias_idx.values())
    pages_since_match = 0
    EARLY_STOP_AFTER_BARREN_PAGES = 20  # stop hunting after 20 unproductive pages
    base = (
        "https://datasets-server.huggingface.co/rows"
        "?dataset=open-llm-leaderboard%2Fcontents"
        "&config=default&split=train"
    )
    log.info("Fetching Open LLM Leaderboard v2 (paginated)…")
    total_rows = 0
    for page in range(OPEN_LLM_MAX_PAGES):
        if not targets_remaining:
            break
        offset = page * OPEN_LLM_PAGE_SIZE
        rows = None
        for attempt in range(OPEN_LLM_RETRIES_PER_PAGE):
            try:
                resp = client.get(
                    f"{base}&offset={offset}&length={OPEN_LLM_PAGE_SIZE}",
                    timeout=30.0,
                )
                resp.raise_for_status()
                rows = resp.json().get("rows", [])
                break
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                if code == 429 and attempt + 1 < OPEN_LLM_RETRIES_PER_PAGE:
                    backoff = 2 ** attempt
                    log.warning("  page %d: 429 rate-limited, sleeping %ds", page, backoff)
                    time.sleep(backoff)
                    continue
                if 500 <= code < 600 and attempt + 1 < OPEN_LLM_RETRIES_PER_PAGE:
                    log.warning("  page %d: %s, retrying", page, code)
                    time.sleep(1)
                    continue
                log.warning("  page %d: %s, skipping", page, code)
                break
            except httpx.HTTPError as e:
                log.warning("  page %d: %s, skipping", page, e)
                break
        time.sleep(0.3)  # gentle throttle to stay under HF rate limits
        if rows is None:
            continue  # transient failure, try next page
        if not rows:
            break
        total_rows += len(rows)
        matched_this_page = 0
        for row in rows:
            r = row.get("row", {})
            repo_id = (r.get("fullname") or r.get("eval_name") or "").lower()
            catalog_id = alias_idx.get(repo_id)
            if not catalog_id:
                continue
            targets_remaining.discard(catalog_id)
            matched_this_page += 1
            for upstream_key, normalized in [
                ("MMLU-PRO", "MMLU-PRO"),
                ("GPQA", "GPQA"),
                ("MATH Lvl 5", "MATH"),
                ("MUSR", "MUSR"),
                ("BBH", "BBH"),
                ("IFEval", "IFEval"),
            ]:
                v = r.get(upstream_key)
                if v is None:
                    continue
                try:
                    out.add(catalog_id, normalized, float(v))
                except (TypeError, ValueError):
                    continue
        if matched_this_page == 0:
            pages_since_match += 1
            if pages_since_match >= EARLY_STOP_AFTER_BARREN_PAGES and out.by_model:
                log.info("  early stop after %d pages with no new matches", pages_since_match)
                break
        else:
            pages_since_match = 0
    log.info("  scanned %d rows, matched %d catalog models", total_rows, len(out.by_model))
    if targets_remaining:
        log.warning("  not found on Open LLM Leaderboard: %s", sorted(targets_remaining))
    return out


# ---------- Source: LiveBench ----------

def fetch_livebench(client: httpx.Client, alias_idx: dict[str, str]) -> RawScores:
    """LiveBench publishes per-month HF datasets whose names rotate
    (`livebench/livebench_2024_11_25` etc.). Rather than hardcode a URL that
    will rot, we discover the most recent dataset via the HF API and pull
    from its rows endpoint. Falls back to an empty result if discovery fails.
    """
    out = RawScores()
    log.info("Fetching LiveBench…")
    # LiveBench publishes per-category HF datasets under livebench/* plus a
    # consolidated livebench/model_judgment. The aggregate view is the easiest
    # to consume; if the rows endpoint is unhappy, skip cleanly.
    rows_url = (
        "https://datasets-server.huggingface.co/rows"
        "?dataset=livebench%2Fmodel_judgment"
        "&config=default&split=train&offset=0&length=100"
    )
    resp = client.get(rows_url, timeout=30.0)
    if resp.status_code != 200:
        log.warning("  LiveBench rows endpoint returned %d; skipping "
                    "(parquet-based fetch is the next step — see README)",
                    resp.status_code)
        return out
    rows = resp.json().get("rows", [])
    log.info("  got %d rows", len(rows))
    for row in rows:
        r = row.get("row", {})
        model = (r.get("model") or r.get("model_name") or "").lower()
        catalog_id = alias_idx.get(model)
        if not catalog_id:
            continue
        for cat in ("coding", "reasoning", "language",
                    "instruction_following", "data_analysis", "mathematics"):
            v = r.get(cat)
            if v is None:
                continue
            try:
                out.add(catalog_id, cat, float(v))
            except (TypeError, ValueError):
                continue
    log.info("  matched %d catalog models", len(out.by_model))
    return out


# ---------- Aggregation ----------

def aggregate(raws: Iterable[RawScores]) -> dict[str, dict[str, float]]:
    """Combine sources -> per-model task scores 0..100.

    Benchmarks are on incompatible scales (IFEval saturates near 80, MATH
    near 30, etc.), so naive averaging crushes good models. We normalize each
    benchmark to its observed max across the catalog (top model -> 100), then
    average per-task. Missing tasks fall back to the overall mean.
    """
    merged: dict[str, dict[str, float]] = {}
    for raw in raws:
        for cid, bench_scores in raw.by_model.items():
            merged.setdefault(cid, {}).update(bench_scores)

    bench_max: dict[str, float] = {}
    for bench_scores in merged.values():
        for bench, score in bench_scores.items():
            if score > bench_max.get(bench, 0.0):
                bench_max[bench] = score

    out: dict[str, dict[str, float]] = {}
    for cid, bench_scores in merged.items():
        per_task: dict[str, list[float]] = {}
        for bench, score in bench_scores.items():
            top = bench_max.get(bench, 0.0)
            if top <= 0:
                continue
            normalized = (score / top) * 100.0
            for task in BENCHMARK_TASKS.get(bench, ()):
                per_task.setdefault(task, []).append(normalized)
        if not per_task:
            continue
        overall = statistics.mean([s for vs in per_task.values() for s in vs])
        scores = {
            task: round(statistics.mean(vs), 1)
            for task, vs in per_task.items()
        }
        for task in ("code", "chat", "reasoning", "general"):
            scores.setdefault(task, round(overall, 1))
        out[cid] = scores
    return out


# ---------- Snapshot assembly ----------

def build_snapshot(scores_by_model: dict[str, dict[str, float]]) -> dict:
    footprints = load_yaml(FOOTPRINTS_PATH)["models"]
    models = []
    skipped = []
    for cid, scores in scores_by_model.items():
        meta = footprints.get(cid)
        if not meta:
            skipped.append(cid)
            continue
        models.append({
            "id": cid,
            "display": meta["display"],
            "params_b": meta["params_b"],
            "footprint_gb": meta["footprint_gb"],
            "ollama": meta.get("ollama"),
            "scores": scores,
        })
    if skipped:
        log.warning("Skipped %d models missing from footprints.yaml: %s",
                    len(skipped), skipped)
    models.sort(key=lambda m: m["scores"].get("general", 0.0), reverse=True)
    return {
        "snapshot_date": date.today().isoformat(),
        "sources": [
            "Open LLM Leaderboard v2 (HuggingFace)",
            "LiveBench",
        ],
        "notes": (
            "Footprints are approximate Q4_K_M GGUF sizes plus ~1GB runtime "
            "overhead. Scores 0-100 are mean-of-benchmark-means per task."
        ),
        "models": models,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    alias_idx = build_alias_index(load_yaml(ALIASES_PATH))

    raws: list[RawScores] = []
    failures: list[str] = []
    with httpx.Client(headers={"User-Agent": "llm-fit-check-refresh"}) as client:
        for name, fn in [
            ("open-llm-leaderboard", fetch_open_llm_leaderboard),
            ("livebench", fetch_livebench),
        ]:
            try:
                raws.append(fn(client, alias_idx))
            except Exception as e:
                log.error("source %s failed: %s", name, e)
                failures.append(name)

    if not raws or all(not r.by_model for r in raws):
        log.error("All sources failed (%s); preserving existing snapshot.", failures)
        return 1

    scores = aggregate(raws)
    if not scores:
        log.error("No models matched any alias; preserving existing snapshot.")
        return 1

    snapshot = build_snapshot(scores)
    log.info("Built snapshot with %d models (%d source failures: %s)",
             len(snapshot["models"]), len(failures), failures)

    if args.dry_run:
        print(json.dumps(snapshot, indent=2))
        return 0

    SNAPSHOT_PATH.write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
    )
    log.info("Wrote %s", SNAPSHOT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
