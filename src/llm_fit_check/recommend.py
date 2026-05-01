from __future__ import annotations

from dataclasses import dataclass

from llm_fit_check.catalog import Catalog, Model, TASKS, load
from llm_fit_check.hardware import Hardware, detect


@dataclass
class Recommendation:
    pick: Model | None
    runners_up: list[Model]
    hardware: Hardware
    catalog: Catalog
    task: str
    fit_budget_gb: float

    def one_liner(self) -> str:
        if not self.pick:
            return (
                f"No open-source LLM in our catalog fits in {self.fit_budget_gb:.1f}GB "
                f"of usable memory. Try llama3.2:1b on CPU, or upgrade hardware."
            )
        m = self.pick
        runtime_hint = f" (`ollama run {m.ollama}`)" if m.ollama else ""
        return (
            f"For {self.task}, run {m.display}{runtime_hint} — "
            f"~{m.footprint_gb:.1f}GB, score {m.score_for(self.task):.0f}/100 "
            f"on the {self.catalog.snapshot_date} leaderboard snapshot."
        )


def _validate_task(task: str) -> str:
    task = task.lower().strip()
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    return task


def recommend(
    task: str = "general",
    *,
    hardware: Hardware | None = None,
    catalog: Catalog | None = None,
    headroom_gb: float = 2.0,
    refresh: bool = False,
    offline: bool = False,
) -> Recommendation:
    """Return the best-fitting open-source LLM for the given task.

    `headroom_gb` reserves memory for KV cache and context. Increase it if you
    plan to use long contexts.
    """
    task = _validate_task(task)
    hw = hardware or detect()
    cat = catalog or load(refresh=refresh, offline=offline)

    budget = max(0.0, hw.usable_memory_gb - headroom_gb)
    fitting = [m for m in cat.models if m.footprint_gb <= budget]
    fitting.sort(key=lambda m: (m.score_for(task), m.params_b), reverse=True)

    pick = fitting[0] if fitting else None
    runners = fitting[1:4]
    return Recommendation(
        pick=pick,
        runners_up=runners,
        hardware=hw,
        catalog=cat,
        task=task,
        fit_budget_gb=budget,
    )
