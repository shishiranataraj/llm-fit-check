from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from llm_fit_check.catalog import TASKS
from llm_fit_check.recommend import recommend

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Tells you which open-source LLM works best on your local setup today.",
)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    task: str = typer.Option(
        "general",
        "--task",
        "-t",
        help=f"What you want to do. One of: {', '.join(TASKS)}.",
    ),
    why: bool = typer.Option(False, "--why", help="Show hardware, runners-up, and source."),
    refresh: bool = typer.Option(False, "--refresh", help="Force-refresh leaderboard cache."),
    offline: bool = typer.Option(False, "--offline", help="Skip network, use cache or bundle."),
    headroom: float = typer.Option(
        2.0, "--headroom", help="GB to reserve for KV cache / context."
    ),
    interactive: bool = typer.Option(
        False, "--ask", help="Ask a couple of questions to pick the task."
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    if interactive:
        task = _ask_task()

    rec = recommend(task=task, headroom_gb=headroom, refresh=refresh, offline=offline)
    console.print(f"[bold green]>[/bold green] {rec.one_liner()}")

    if why:
        _print_details(rec)


@app.command()
def hardware() -> None:
    """Print detected hardware."""
    from llm_fit_check.hardware import detect

    hw = detect()
    table = Table(title="Detected hardware", show_header=False)
    table.add_row("OS", hw.os)
    table.add_row("CPU cores", str(hw.cpu_cores))
    table.add_row("RAM", f"{hw.ram_gb} GB")
    table.add_row("GPU", hw.gpu_name or "—")
    table.add_row("VRAM", f"{hw.vram_gb} GB" if hw.vram_gb else "—")
    table.add_row("Runtime", hw.runtime)
    table.add_row("Usable memory for inference", f"{hw.usable_memory_gb} GB")
    console.print(table)


@app.command(name="list")
def list_models(
    task: str = typer.Option("general", "--task", "-t"),
    offline: bool = typer.Option(False, "--offline"),
) -> None:
    """List the catalog ranked for a task."""
    rec = recommend(task=task, offline=offline)
    table = Table(title=f"Catalog ranked for: {task}  (snapshot {rec.catalog.snapshot_date})")
    table.add_column("Model")
    table.add_column("Footprint", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Fits?", justify="center")
    budget = rec.fit_budget_gb
    ranked = sorted(
        rec.catalog.models, key=lambda m: m.score_for(task), reverse=True
    )
    for m in ranked:
        fits = "[green]yes[/green]" if m.footprint_gb <= budget else "[red]no[/red]"
        table.add_row(m.display, f"{m.footprint_gb:.1f} GB", f"{m.score_for(task):.0f}", fits)
    console.print(table)
    console.print(f"[dim]Memory budget after headroom: {budget:.1f} GB[/dim]")


def _ask_task() -> str:
    console.print("[bold]What do you mostly want to do?[/bold]")
    options = {
        "1": ("code", "Write/refactor code"),
        "2": ("chat", "General conversation, drafting, summarization"),
        "3": ("reasoning", "Math, logic, multi-step problems"),
        "4": ("general", "A bit of everything"),
    }
    for k, (_, label) in options.items():
        console.print(f"  {k}) {label}")
    choice = typer.prompt("Pick 1-4", default="4")
    return options.get(choice, ("general", ""))[0]


def _print_details(rec) -> None:
    hw = rec.hardware
    console.print()
    console.print(
        f"[dim]Hardware:[/dim] {hw.gpu_name or 'CPU only'} · "
        f"{hw.usable_memory_gb} GB usable · runtime={hw.runtime}"
    )
    console.print(
        f"[dim]Catalog:[/dim] snapshot {rec.catalog.snapshot_date} "
        f"({rec.catalog.origin}) · sources: {', '.join(rec.catalog.sources)}"
    )
    if rec.runners_up:
        console.print("[dim]Runners-up:[/dim]")
        for m in rec.runners_up:
            console.print(
                f"  · {m.display} — {m.footprint_gb:.1f} GB, "
                f"score {m.score_for(rec.task):.0f}"
            )


if __name__ == "__main__":
    app()
