from llm_fit_check.catalog import load
from llm_fit_check.hardware import Hardware
from llm_fit_check.recommend import recommend


def _hw(usable: float) -> Hardware:
    # Build a Hardware with controlled usable_memory_gb via VRAM (non-unified).
    return Hardware(
        os="Linux",
        cpu_cores=8,
        ram_gb=64.0,
        gpu_vendor="nvidia",
        gpu_name="Test GPU",
        vram_gb=usable,
        unified_memory=False,
    )


def test_loads_bundled_catalog_offline():
    cat = load(offline=True)
    assert cat.models
    assert cat.snapshot_date


def test_recommends_within_budget():
    cat = load(offline=True)
    rec = recommend(task="code", hardware=_hw(8.0), catalog=cat, headroom_gb=2.0)
    assert rec.pick is not None
    assert rec.pick.footprint_gb <= 6.0
    assert "score" in rec.one_liner()


def test_no_fit_returns_none():
    cat = load(offline=True)
    rec = recommend(task="general", hardware=_hw(0.5), catalog=cat, headroom_gb=2.0)
    assert rec.pick is None
    assert "No open-source LLM" in rec.one_liner()


def test_picks_top_score_when_plenty_of_memory():
    cat = load(offline=True)
    rec = recommend(task="reasoning", hardware=_hw(80.0), catalog=cat)
    # Top reasoning model in snapshot is DeepSeek-R1-Distill 32B at 88.
    assert rec.pick.scores["reasoning"] >= 85
