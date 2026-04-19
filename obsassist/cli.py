"""CLI entry-point for the obsassist tool.

Commands
--------
obsassist analyze  --file <path> [--yes] [--config <path>]
    Full analysis: Summary + Questions + Metadata suggestions.

obsassist metadata --file <path> [--yes] [--config <path>]
    Metadata suggestions only (Summary / Questions are preserved when a
    block already exists).

obsassist index build [--config <path>] [--index-path <path>]
    Full rebuild of the full-text search index from scratch.

obsassist index update [--config <path>] [--index-path <path>]
    Incremental index update (add/update changed files, remove deleted ones).

obsassist search "<query>" [--config <path>] [--index-path <path>] [--limit N]
    Full-text search across all indexed notes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from .config import get_index_path, load_config
from .diff import generate_diff
from .filters import is_excluded
from .indexer import build_index, update_index
from .ollama_client import OllamaClient
from .parser import (
    AssistantBlock,
    find_assistant_section,
    parse_existing_block,
    update_note,
)
from .prompts import (
    build_analyze_prompt,
    build_metadata_prompt,
    parse_ollama_response,
)
from .search import search as fts_search

console = Console()


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """Obsidian PARA notes AI assistant powered by Ollama."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_note(file_path: str) -> tuple[Path, str]:
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    return path, content


def _check_exclusion(path: Path, config) -> None:
    vault_root = Path(config.vault_root) if config.vault_root else None
    target = path.resolve()
    root = vault_root.resolve() if vault_root is not None else None
    if is_excluded(target, root, config.exclude_paths):
        console.print(
            f"[red]Error:[/red] {path} is inside an excluded path "
            f"({', '.join(config.exclude_paths[:3])}…). Aborting."
        )
        sys.exit(1)


def _build_client(config) -> OllamaClient:
    return OllamaClient(
        base_url=config.ollama.base_url,
        model=config.ollama.model,
        temperature=config.ollama.temperature,
    )


def _show_and_confirm(diff: str, path: Path, updated: str, yes: bool) -> None:
    """Display diff, optionally prompt, then write changes."""
    if not diff:
        console.print("[green]No changes needed.[/green]")
        return

    console.print("\n[bold]Proposed changes:[/bold]")
    console.print(Syntax(diff, "diff", theme="monokai", line_numbers=False))

    if not yes:
        if not click.confirm("\nApply changes?"):
            console.print("[yellow]Aborted — no changes written.[/yellow]")
            return

    path.write_text(updated, encoding="utf-8")
    console.print(f"\n[green]✓ Updated:[/green] {path}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--file",
    "-f",
    "file_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the Obsidian note (.md file).",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Auto-apply changes without confirmation.",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to a YAML config file (default: config.yml in cwd).",
)
def analyze(file_path: str, yes: bool, config_path: str | None) -> None:
    """Analyze a note: generate Summary, Questions, and Metadata suggestions."""
    cfg = load_config(Path(config_path) if config_path else None)
    path, content = _load_note(file_path)
    _check_exclusion(path, cfg)

    client = _build_client(cfg)
    if not client.health_check():
        console.print(
            f"[red]Error:[/red] Cannot reach Ollama at [bold]{cfg.ollama.base_url}[/bold].\n"
            "Make sure Ollama is running: [bold]ollama serve[/bold]"
        )
        sys.exit(1)

    console.print(f"[cyan]Analyzing:[/cyan] {path.name}  "
                  f"[dim]model={cfg.ollama.model}[/dim]")

    with console.status("[yellow]Calling Ollama…[/yellow]"):
        response = client.generate(build_analyze_prompt(content))

    new_block = parse_ollama_response(response)
    updated = update_note(content, new_block)
    diff = generate_diff(content, updated, path.name)
    _show_and_confirm(diff, path, updated, yes)


@main.command()
@click.option(
    "--file",
    "-f",
    "file_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the Obsidian note (.md file).",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Auto-apply changes without confirmation.",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to a YAML config file (default: config.yml in cwd).",
)
def metadata(file_path: str, yes: bool, config_path: str | None) -> None:
    """Generate metadata suggestions for a note (preserves existing Summary/Questions)."""
    cfg = load_config(Path(config_path) if config_path else None)
    path, content = _load_note(file_path)
    _check_exclusion(path, cfg)

    client = _build_client(cfg)
    if not client.health_check():
        console.print(
            f"[red]Error:[/red] Cannot reach Ollama at [bold]{cfg.ollama.base_url}[/bold].\n"
            "Make sure Ollama is running: [bold]ollama serve[/bold]"
        )
        sys.exit(1)

    console.print(f"[cyan]Generating metadata for:[/cyan] {path.name}  "
                  f"[dim]model={cfg.ollama.model}[/dim]")

    with console.status("[yellow]Calling Ollama…[/yellow]"):
        response = client.generate(build_metadata_prompt(content))

    new_block = parse_ollama_response(response)

    # Preserve existing Summary and Questions if a block is already present.
    pos = find_assistant_section(content)
    if pos is not None:
        existing = parse_existing_block(content[pos[0] : pos[1]])
        new_block = AssistantBlock(
            summary=existing.summary,
            questions=existing.questions,
            metadata_yaml=new_block.metadata_yaml,
        )

    updated = update_note(content, new_block)
    diff = generate_diff(content, updated, path.name)
    _show_and_confirm(diff, path, updated, yes)


# ---------------------------------------------------------------------------
# Shared index options
# ---------------------------------------------------------------------------

_config_option = click.option(
    "--config",
    "-c",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to a YAML config file (default: config.yml in cwd).",
)

_index_path_option = click.option(
    "--index-path",
    "index_path_override",
    default=None,
    type=click.Path(),
    help=(
        "Path to the SQLite index file. "
        "Overrides config and the platform default."
    ),
)


def _resolve_index_path(cfg, index_path_override: str | None) -> Path:
    if index_path_override:
        return Path(index_path_override)
    return get_index_path(cfg)


# ---------------------------------------------------------------------------
# Index command group
# ---------------------------------------------------------------------------


@main.group()
def index() -> None:
    """Build and update the full-text search index."""


@index.command("build")
@_config_option
@_index_path_option
def index_build(config_path: str | None, index_path_override: str | None) -> None:
    """Full rebuild of the index from scratch."""
    cfg = load_config(Path(config_path) if config_path else None)

    if not cfg.vault_root:
        console.print(
            "[red]Error:[/red] vault_root is not set. "
            "Add it to your config.yml or set --config."
        )
        sys.exit(1)

    vault_root = Path(cfg.vault_root)
    if not vault_root.is_dir():
        console.print(f"[red]Error:[/red] vault_root does not exist: {vault_root}")
        sys.exit(1)

    idx_path = _resolve_index_path(cfg, index_path_override)
    console.print(f"[cyan]Building index:[/cyan] {idx_path}")
    console.print(f"[dim]Vault: {vault_root}[/dim]")

    with console.status("[yellow]Indexing…[/yellow]"):
        indexed, skipped = build_index(vault_root, idx_path, cfg)

    console.print(
        f"[green]✓ Done.[/green] Indexed: [bold]{indexed}[/bold]"
        + (f"  Skipped (read errors): {skipped}" if skipped else "")
    )


@index.command("update")
@_config_option
@_index_path_option
def index_update(config_path: str | None, index_path_override: str | None) -> None:
    """Incremental update: re-index changed files, remove deleted ones."""
    cfg = load_config(Path(config_path) if config_path else None)

    if not cfg.vault_root:
        console.print(
            "[red]Error:[/red] vault_root is not set. "
            "Add it to your config.yml or set --config."
        )
        sys.exit(1)

    vault_root = Path(cfg.vault_root)
    if not vault_root.is_dir():
        console.print(f"[red]Error:[/red] vault_root does not exist: {vault_root}")
        sys.exit(1)

    idx_path = _resolve_index_path(cfg, index_path_override)
    console.print(f"[cyan]Updating index:[/cyan] {idx_path}")
    console.print(f"[dim]Vault: {vault_root}[/dim]")

    with console.status("[yellow]Scanning for changes…[/yellow]"):
        added_or_updated, deleted, skipped = update_index(vault_root, idx_path, cfg)

    parts = [f"Updated: [bold]{added_or_updated}[/bold]"]
    if deleted:
        parts.append(f"Removed: {deleted}")
    if skipped:
        parts.append(f"Skipped (read errors): {skipped}")
    console.print("[green]✓ Done.[/green]  " + "  ".join(parts))


# ---------------------------------------------------------------------------
# Search command
# ---------------------------------------------------------------------------


@main.command()
@click.argument("query")
@_config_option
@_index_path_option
@click.option(
    "--limit",
    "-n",
    default=20,
    show_default=True,
    help="Maximum number of results.",
)
def search(
    query: str,
    config_path: str | None,
    index_path_override: str | None,
    limit: int,
) -> None:
    """Full-text search across indexed notes.

    QUERY is a plain-text search expression (FTS5 syntax is also supported).
    """
    cfg = load_config(Path(config_path) if config_path else None)
    idx_path = _resolve_index_path(cfg, index_path_override)

    if not idx_path.exists():
        console.print(
            f"[red]Error:[/red] Index not found at {idx_path}. "
            "Run [bold]obsassist index build[/bold] first."
        )
        sys.exit(1)

    results = fts_search(idx_path, query, limit=limit)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    table.add_column("#", style="dim", width=3, no_wrap=True)
    table.add_column("File", style="green", no_wrap=False)
    table.add_column("Title", no_wrap=False)
    table.add_column("Snippet", style="dim", no_wrap=False)

    for i, result in enumerate(results, 1):
        table.add_row(
            str(i),
            result.path,
            result.title,
            result.snippet,
        )

    console.print(table)
    console.print(f"\n[dim]{len(results)} result(s) for:[/dim] {query}")

