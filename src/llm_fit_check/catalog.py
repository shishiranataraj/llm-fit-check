from __future__ import annotations

import json
import time
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import httpx
from platformdirs import user_cache_dir

REMOTE_URL = (
    "https://raw.githubusercontent.com/your-org/llm-fit-check/"
    "main/src/llm_fit_check/data/snapshot.json"
)
CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # match the weekly upstream refresh
TASKS = ("code", "chat", "reasoning", "general")


@dataclass(frozen=True)
class Model:
    id: str
    display: str
    params_b: float
    footprint_gb: float
    ollama: str | None
    scores: dict[str, float]

    def score_for(self, task: str) -> float:
        return float(self.scores.get(task, self.scores.get("general", 0.0)))


@dataclass(frozen=True)
class Catalog:
    snapshot_date: str
    sources: list[str]
    models: list[Model]
    origin: str  # "remote", "cache", or "bundled"


def _bundled_path() -> Path:
    return Path(str(files("llm_fit_check.data").joinpath("snapshot.json")))


def _cache_path() -> Path:
    base = Path(user_cache_dir("llm-fit-check"))
    base.mkdir(parents=True, exist_ok=True)
    return base / "snapshot.json"


def _parse(payload: dict, origin: str) -> Catalog:
    models = [
        Model(
            id=m["id"],
            display=m["display"],
            params_b=float(m["params_b"]),
            footprint_gb=float(m["footprint_gb"]),
            ollama=m.get("ollama"),
            scores={k: float(v) for k, v in m["scores"].items()},
        )
        for m in payload["models"]
    ]
    return Catalog(
        snapshot_date=payload.get("snapshot_date", "unknown"),
        sources=list(payload.get("sources", [])),
        models=models,
        origin=origin,
    )


def load(refresh: bool = False, offline: bool = False) -> Catalog:
    cache = _cache_path()

    if not offline and (refresh or _stale(cache)):
        try:
            with httpx.Client(timeout=5.0, follow_redirects=True) as client:
                resp = client.get(REMOTE_URL)
                resp.raise_for_status()
                cache.write_text(resp.text, encoding="utf-8")
                return _parse(resp.json(), origin="remote")
        except Exception:
            pass  # fall through to cache/bundled

    if cache.exists():
        try:
            return _parse(json.loads(cache.read_text(encoding="utf-8")), origin="cache")
        except Exception:
            pass

    return _parse(json.loads(_bundled_path().read_text(encoding="utf-8")), origin="bundled")


def _stale(cache: Path) -> bool:
    if not cache.exists():
        return True
    age = time.time() - cache.stat().st_mtime
    return age > CACHE_TTL_SECONDS
