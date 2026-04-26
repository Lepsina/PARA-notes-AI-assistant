"""CLI entry-point for the obsassist tool.

Commands
--------
obsassist analyze  --file <path> [--yes] [--config <path>]
    Full analysis: Summary + Questions + Metadata suggestions.

obsassist metadata update --file <path> [--yes] [--config <path>]
    Metadata suggestions only for a single note (Summary / Questions are
    preserved when a block already exists).

obsassist metadata apply --tag <tag> [--vault <path>] [--remove-tag]
                         [--dry-run] [--diff] [--limit N] [--yes] [--force]
                         [--config <path>]
    Batch-apply metadata to all vault notes that contain the marker tag
    (in frontmatter tags list or as an inline #tag in the body).

obsassist index build [--config <path>] [--index-path <path>]
    Full rebuild of the full-text search index from scratch.

obsassist index update [--config <path>] [--index-path <path>]
    Incremental index update (add/update changed files, remove deleted ones).

obsassist search "<query>" [--config <path>] [--index-path <path>] [--limit N]
    Full-text search across all indexed notes.

obsassist embeddings build [--config <path>] [--index-path <path>]
    Full rebuild of the embeddings index (chunks + vectors).

obsassist embeddings update [--config <path>] [--index-path <path>]
    Incremental embeddings update (changed files only).

obsassist ask "<question>" [--mode fts|vector|hybrid] [--k N]
             [--candidates N] [--config <path>] [--index-path <path>]
             [--save-to <path>]
    Answer a question using retrieved vault chunks.  Includes a Sources section.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table

from .config import get_index_path, load_config
from .diff import generate_diff
from .embeddings import build_embeddings, update_embeddings
from .filters import is_excluded
from .indexer import build_index, update_index
from .metadata_guard import ALLOWED_KEYS, apply_metadata_to_content, load_vocab
from .tag_scanner import has_marker_tag, remove_marker_tag
from .ollama_client import OllamaClient
from .parser import (
    AssistantBlock,
    find_assistant_section,
    parse_existing_block,
    update_note,
)
from .prompts import (
    build_analyze_prompt,
    build_ask_prompt,
    build_frontmatter_prompt,
    build_metadata_prompt,
    parse_ollama_response,
)
from .retrieval import (
    ChunkResult,
    retrieve_fts,
    retrieve_hybrid,
    retrieve_vector,
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
# Metadata apply helpers
# ---------------------------------------------------------------------------


def _metadata_state_path(vault_root: Path) -> Path:
    """Return the path to the metadata-apply state file."""
    return vault_root / ".obsassist" / "metadata-apply-state.json"


def _load_processed_set(state_path: Path, tag: str) -> set[str]:
    """Return the set of already-processed absolute file paths for *tag*."""
    if not state_path.exists():
        return set()
    try:
        with open(state_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("tag") == tag:
            return set(data.get("processed", []))
    except (json.JSONDecodeError, OSError):
        pass
    return set()


def _save_processed_set(state_path: Path, tag: str, processed: set[str]) -> None:
    """Persist the set of processed file paths to the state file."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"tag": tag, "processed": sorted(processed)},
            fh,
            indent=2,
            ensure_ascii=False,
        )


def _write_file(path: Path, content: str, *, make_backup: bool = False) -> None:
    """Write *content* to *path*, optionally creating a ``.bak`` copy first."""
    if make_backup:
        bak = path.with_suffix(path.suffix + ".bak")
        bak.write_bytes(path.read_bytes())
    path.write_text(content, encoding="utf-8")


def _llm_process_file(
    md_path: Path,
    client: OllamaClient,
    allowed: set[str],
    vocab: dict | None,
    effective_force: bool,
    marker_tag: str,
    do_remove_tag: bool,
) -> tuple[str, str, bool, str | None]:
    """Read a note, call the LLM, and compute new content.

    Returns ``(original, new_content, changed, error)``.
    *error* is ``None`` on success, or a short error description.
    """
    try:
        original = md_path.read_text(encoding="utf-8")
    except OSError as exc:
        return "", "", False, f"read error: {exc}"

    try:
        response = client.generate(build_frontmatter_prompt(original))
    except Exception as exc:  # noqa: BLE001
        return original, original, False, f"LLM error: {exc}"

    try:
        new_content, changed = apply_metadata_to_content(
            original,
            response,
            allowed_keys=allowed,
            vocab=vocab,
            force=effective_force,
        )
    except ValueError as exc:
        return original, original, False, f"validation error: {exc}"

    if do_remove_tag:
        stripped = remove_marker_tag(new_content, marker_tag)
        if stripped != new_content:
            new_content = stripped
            changed = True

    return original, new_content, changed, None


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


# ---------------------------------------------------------------------------
# Metadata command group
# ---------------------------------------------------------------------------


@main.group()
def metadata() -> None:
    """Update YAML frontmatter metadata — body content is never modified."""


@metadata.command("update")
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
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing frontmatter fields (full regeneration).",
)
def metadata_update(file_path: str, yes: bool, config_path: str | None, force: bool) -> None:
    """Update YAML frontmatter metadata for a single note.

    LLM suggestions are validated against a strict schema allowlist before
    writing.  Unknown keys are silently dropped.  Existing user-authored
    values are preserved unless --force is used.
    """
    cfg = load_config(Path(config_path) if config_path else None)
    path, content = _load_note(file_path)
    _check_exclusion(path, cfg)

    # Determine which model to use: dedicated metadata model → fallback to ask
    metadata_model = cfg.metadata.model or cfg.ollama.model
    effective_force = force or cfg.metadata.force

    client = OllamaClient(
        base_url=cfg.ollama.base_url,
        model=metadata_model,
        temperature=cfg.ollama.temperature,
    )
    if not client.health_check():
        console.print(
            f"[red]Error:[/red] Cannot reach Ollama at [bold]{cfg.ollama.base_url}[/bold].\n"
            "Make sure Ollama is running: [bold]ollama serve[/bold]"
        )
        sys.exit(1)

    # Load vocab for normalisation
    vocab = load_vocab(cfg.metadata.vocab_path) if cfg.metadata.vocab_path else None

    # Build the effective allowed-keys set
    allowed = set(ALLOWED_KEYS) | set(cfg.metadata.extra_allowed_keys)

    console.print(
        f"[cyan]Updating metadata for:[/cyan] {path.name}  "
        f"[dim]model={metadata_model}[/dim]"
    )

    with console.status("[yellow]Calling Ollama…[/yellow]"):
        response = client.generate(build_frontmatter_prompt(content))

    try:
        updated, changed = apply_metadata_to_content(
            content,
            response,
            allowed_keys=allowed,
            vocab=vocab,
            force=effective_force,
        )
    except ValueError as exc:
        console.print(f"[red]Metadata validation error:[/red] {exc}")
        console.print("[yellow]No changes written — note is unchanged.[/yellow]")
        return

    if not changed:
        console.print("[green]No changes needed.[/green]")
        return

    diff = generate_diff(content, updated, path.name)
    _show_and_confirm(diff, path, updated, yes)


@metadata.command("apply")
@click.option(
    "--tag",
    "-t",
    "marker_tag",
    required=True,
    help="Marker tag to look for (e.g. add-metadata).  Leading '#' is optional.",
)
@click.option(
    "--remove-tag",
    "remove_tag",
    is_flag=True,
    default=False,
    help="Remove the marker tag from the note after a successful metadata update.",
)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help="Scan and show proposed changes without writing any files.",
)
@click.option(
    "--diff",
    "show_diff",
    is_flag=True,
    default=False,
    help="Show a unified diff of proposed frontmatter changes.",
)
@click.option(
    "--limit",
    "-n",
    default=0,
    show_default=True,
    help="Maximum number of notes to process (0 = no limit).",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Auto-apply changes without per-file confirmation.",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to a YAML config file (default: config.yml in cwd).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing frontmatter fields (full regeneration).",
)
@click.option(
    "--batch",
    "--batch-size",
    "batch_size",
    default=0,
    show_default=True,
    help="Print a progress line after every N files (0 = disabled).",
)
@click.option(
    "--workers",
    "-w",
    default=1,
    show_default=True,
    help=(
        "Number of parallel LLM workers.  "
        "Requires --yes or --dry-run; interactive mode falls back to 1."
    ),
)
@click.option(
    "--path",
    "scope_path",
    default=None,
    type=click.Path(),
    help="Restrict the scan to this sub-path within the vault.",
)
@click.option(
    "--resume",
    "resume",
    is_flag=True,
    default=False,
    help=(
        "Skip files already processed in a previous run "
        "(state stored in <vault>/.obsassist/metadata-apply-state.json)."
    ),
)
@click.option(
    "--backup/--no-backup",
    "backup",
    default=True,
    help="Create a .bak copy before overwriting each note (default: on).",
)
def metadata_apply(
    marker_tag: str,
    remove_tag: bool,
    dry_run: bool,
    show_diff: bool,
    limit: int,
    yes: bool,
    config_path: str | None,
    force: bool,
    batch_size: int,
    workers: int,
    scope_path: str | None,
    resume: bool,
    backup: bool,
) -> None:
    """Batch-apply metadata to notes marked with a tag.

    Scans the vault for all Markdown notes that contain MARKER_TAG either in
    their frontmatter ``tags`` list or as an inline ``#tag`` in the body.
    For each matching note the LLM is called and metadata is updated using
    the same strict guardrails as ``metadata update``.

    Examples:

    \b
      # Preview changes without writing
      obsassist metadata apply --tag add-metadata --dry-run --diff

    \b
      # Apply to all marked notes, auto-confirm, remove the marker afterwards
      obsassist metadata apply --tag add-metadata --yes --remove-tag

    \b
      # Process at most 5 notes
      obsassist metadata apply --tag add-metadata --limit 5

    \b
      # Resume a previously interrupted batch
      obsassist metadata apply --tag add-metadata --yes --resume
    """
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

    # Resolve scan root (optional sub-path restriction)
    if scope_path:
        scan_root = vault_root / scope_path
        if not scan_root.is_dir():
            console.print(
                f"[red]Error:[/red] --path '{scope_path}' does not exist "
                f"inside vault_root: {vault_root}"
            )
            sys.exit(1)
    else:
        scan_root = vault_root

    # Determine model and effective options
    metadata_model = cfg.metadata.model or cfg.ollama.model
    effective_force = force or cfg.metadata.force

    client = OllamaClient(
        base_url=cfg.ollama.base_url,
        model=metadata_model,
        temperature=cfg.ollama.temperature,
    )
    if not client.health_check():
        console.print(
            f"[red]Error:[/red] Cannot reach Ollama at [bold]{cfg.ollama.base_url}[/bold].\n"
            "Make sure Ollama is running: [bold]ollama serve[/bold]"
        )
        sys.exit(1)

    vocab = load_vocab(cfg.metadata.vocab_path) if cfg.metadata.vocab_path else None
    allowed = set(ALLOWED_KEYS) | set(cfg.metadata.extra_allowed_keys)
    vault_root_resolved = vault_root.resolve()

    # Interactive mode is incompatible with workers > 1
    effective_workers = workers
    if effective_workers > 1 and not (yes or dry_run):
        console.print(
            "[yellow]Warning:[/yellow] --workers > 1 requires --yes or --dry-run; "
            "falling back to sequential processing."
        )
        effective_workers = 1

    # ------------------------------------------------------------------
    # Load resume state
    # ------------------------------------------------------------------
    state_file = _metadata_state_path(vault_root)
    processed_set: set[str] = (
        _load_processed_set(state_file, marker_tag) if resume else set()
    )

    # ------------------------------------------------------------------
    # Scan vault for marked files
    # ------------------------------------------------------------------
    console.print(
        f"[cyan]Scanning vault:[/cyan] {scan_root}  "
        f"[dim]tag={marker_tag}[/dim]"
    )

    discovered: list[Path] = []
    with console.status("[yellow]Scanning…[/yellow]"):
        for md_path in sorted(scan_root.rglob("*.md")):
            if is_excluded(md_path.resolve(), vault_root_resolved, cfg.exclude_paths):
                continue
            try:
                content = md_path.read_text(encoding="utf-8")
            except OSError:
                continue
            if has_marker_tag(content, marker_tag):
                discovered.append(md_path)

    n_discovered = len(discovered)
    if not discovered:
        console.print(f"[yellow]No notes found containing tag '{marker_tag}'.[/yellow]")
        return

    # Apply --limit before resume filtering
    selected = discovered[:limit] if limit and limit > 0 else list(discovered)
    n_selected = len(selected)

    # Skip already-processed files when resuming
    if resume and processed_set:
        selected = [p for p in selected if str(p.resolve()) not in processed_set]
        n_skipped_resume = n_selected - len(selected)
        if n_skipped_resume:
            console.print(
                f"[dim]--resume: skipping {n_skipped_resume} "
                f"already-processed file(s).[/dim]"
            )

    console.print(
        f"[green]Found {n_discovered} note(s)[/green] "
        f"containing tag '[bold]{marker_tag}[/bold]'"
        + (f", selected {n_selected}" if n_selected != n_discovered else "")
        + (f", processing {len(selected)}" if len(selected) != n_selected else "")
        + "."
    )
    if dry_run:
        console.print("[dim]Dry-run — no files will be written.[/dim]")

    # ------------------------------------------------------------------
    # Process files (sequential or parallel LLM calls)
    # ------------------------------------------------------------------
    n_processed = n_updated = n_skipped = n_errors = 0

    def _handle_result(
        md_path: Path,
        original: str,
        new_content: str,
        changed: bool,
        error: str | None,
    ) -> None:
        nonlocal n_processed, n_updated, n_skipped, n_errors

        if error:
            console.print(f"[red]Error ({md_path.name}):[/red] {error}")
            n_errors += 1
            return

        n_processed += 1

        if not changed:
            console.print(f"[green]  No changes needed:[/green] {md_path.name}")
            n_skipped += 1
            return

        # Show diff when requested or in interactive mode
        needs_diff = show_diff or (not yes and not dry_run)
        if needs_diff:
            diff_text = generate_diff(original, new_content, md_path.name)
            if diff_text:
                console.print(f"\n[bold]Proposed changes:[/bold] {md_path.name}")
                console.print(
                    Syntax(diff_text, "diff", theme="monokai", line_numbers=False)
                )

        if dry_run:
            console.print(f"[dim]  (dry-run) Would update: {md_path.name}[/dim]")
            n_updated += 1
            return

        if yes:
            _write_file(md_path, new_content, make_backup=backup)
            console.print(f"[green]  ✓ Updated:[/green] {md_path.name}")
            n_updated += 1
            processed_set.add(str(md_path.resolve()))
            _save_processed_set(state_file, marker_tag, processed_set)
        else:
            if not needs_diff:
                diff_text = generate_diff(original, new_content, md_path.name)
                if diff_text:
                    console.print(f"\n[bold]Proposed changes:[/bold] {md_path.name}")
                    console.print(
                        Syntax(diff_text, "diff", theme="monokai", line_numbers=False)
                    )
            if click.confirm(f"\nApply changes to {md_path.name}?"):
                _write_file(md_path, new_content, make_backup=backup)
                console.print(f"[green]  ✓ Updated:[/green] {md_path.name}")
                n_updated += 1
                processed_set.add(str(md_path.resolve()))
                _save_processed_set(state_file, marker_tag, processed_set)
            else:
                console.print("[yellow]  Skipped.[/yellow]")
                n_skipped += 1

    if effective_workers <= 1:
        # Sequential processing
        for i, md_path in enumerate(selected):
            console.print(
                f"\n[cyan]Processing:[/cyan] {md_path.name}  "
                f"[dim]model={metadata_model}[/dim]"
            )
            with console.status("[yellow]Calling Ollama…[/yellow]"):
                result = _llm_process_file(
                    md_path, client, allowed, vocab, effective_force,
                    marker_tag, remove_tag,
                )
            _handle_result(md_path, *result)
            if batch_size > 0 and (i + 1) % batch_size == 0:
                console.print(
                    f"[dim]  Batch progress: {i + 1}/{len(selected)} files[/dim]"
                )
    else:
        # Parallel LLM calls; results handled sequentially from the main thread
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = [
                pool.submit(
                    _llm_process_file,
                    md_path, client, allowed, vocab, effective_force,
                    marker_tag, remove_tag,
                )
                for md_path in selected
            ]
            for i, (md_path, future) in enumerate(zip(selected, futures)):
                console.print(
                    f"\n[cyan]Processing:[/cyan] {md_path.name}  "
                    f"[dim]model={metadata_model}[/dim]"
                )
                _handle_result(md_path, *future.result())
                if batch_size > 0 and (i + 1) % batch_size == 0:
                    console.print(
                        f"[dim]  Batch progress: {i + 1}/{len(selected)} files[/dim]"
                    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary_parts = [
        f"Discovered: [bold]{n_discovered}[/bold]",
        f"Processed: [bold]{n_processed}[/bold]",
        f"Updated: [bold]{n_updated}[/bold]",
    ]
    if n_selected != n_discovered:
        summary_parts.insert(1, f"Selected: [bold]{n_selected}[/bold]")
    if n_skipped:
        summary_parts.append(f"Skipped: {n_skipped}")
    if n_errors:
        summary_parts.append(f"Errors: [red]{n_errors}[/red]")
    console.print("\n[green]✓ Done.[/green]  " + "  ".join(summary_parts))


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


# ---------------------------------------------------------------------------
# Embeddings command group
# ---------------------------------------------------------------------------


@main.group()
def embeddings() -> None:
    """Build and update the semantic embeddings index."""


@embeddings.command("build")
@_config_option
@_index_path_option
def embeddings_build(config_path: str | None, index_path_override: str | None) -> None:
    """Full rebuild of the embeddings index (chunks + vectors)."""
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
    emb_client = OllamaClient(
        base_url=cfg.embeddings.base_url,
        model=cfg.embeddings.model,
    )

    if not emb_client.health_check():
        console.print(
            f"[red]Error:[/red] Cannot reach Ollama at "
            f"[bold]{cfg.embeddings.base_url}[/bold].\n"
            "Make sure Ollama is running: [bold]ollama serve[/bold]\n"
            "Then pull the embedding model: "
            f"[bold]ollama pull {cfg.embeddings.model}[/bold]"
        )
        sys.exit(1)

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return [emb_client.embed(t) for t in texts]

    console.print(f"[cyan]Building embeddings index:[/cyan] {idx_path}")
    console.print(
        f"[dim]Vault: {vault_root}  "
        f"model: {cfg.embeddings.model}  "
        f"chunk_size: {cfg.embeddings.chunk_size}[/dim]"
    )

    with console.status("[yellow]Embedding chunks…[/yellow]"):
        embedded, skipped = build_embeddings(vault_root, idx_path, cfg, embed_fn)

    console.print(
        f"[green]✓ Done.[/green] Embedded: [bold]{embedded}[/bold]"
        + (f"  Skipped (errors): {skipped}" if skipped else "")
    )


@embeddings.command("update")
@_config_option
@_index_path_option
def embeddings_update(config_path: str | None, index_path_override: str | None) -> None:
    """Incremental embeddings update: re-embed changed files, remove deleted ones."""
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
    emb_client = OllamaClient(
        base_url=cfg.embeddings.base_url,
        model=cfg.embeddings.model,
    )

    if not emb_client.health_check():
        console.print(
            f"[red]Error:[/red] Cannot reach Ollama at "
            f"[bold]{cfg.embeddings.base_url}[/bold].\n"
            "Make sure Ollama is running: [bold]ollama serve[/bold]\n"
            "Then pull the embedding model: "
            f"[bold]ollama pull {cfg.embeddings.model}[/bold]"
        )
        sys.exit(1)

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return [emb_client.embed(t) for t in texts]

    console.print(f"[cyan]Updating embeddings index:[/cyan] {idx_path}")
    console.print(f"[dim]Vault: {vault_root}  model: {cfg.embeddings.model}[/dim]")

    with console.status("[yellow]Scanning for changes…[/yellow]"):
        updated, deleted, skipped = update_embeddings(vault_root, idx_path, cfg, embed_fn)

    parts = [f"Updated: [bold]{updated}[/bold]"]
    if deleted:
        parts.append(f"Removed: {deleted}")
    if skipped:
        parts.append(f"Skipped (errors): {skipped}")
    console.print("[green]✓ Done.[/green]  " + "  ".join(parts))


# ---------------------------------------------------------------------------
# Ask command
# ---------------------------------------------------------------------------

_MAX_CONTEXT_CHARS = 8000


def _build_context(chunks: list[ChunkResult], max_chars: int = _MAX_CONTEXT_CHARS) -> str:
    """Build a context string from *chunks*, respecting the *max_chars* budget."""
    parts: list[str] = []
    total = 0
    for c in chunks:
        heading_prefix = f"[{c.heading}] " if c.heading else ""
        entry = f"--- {c.path} {heading_prefix}---\n{c.content}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n".join(parts)


def _format_sources(chunks: list[ChunkResult]) -> str:
    """Return a markdown Sources section from retrieved *chunks*."""
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for c in chunks:
        key = (c.path, c.heading)
        if key in seen:
            continue
        seen.add(key)
        if c.heading:
            lines.append(f"- {c.path} — {c.heading}")
        else:
            lines.append(f"- {c.path}")
    return "\n".join(lines)


@main.command()
@click.argument("question")
@_config_option
@_index_path_option
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["fts", "vector", "hybrid"], case_sensitive=False),
    default="hybrid",
    show_default=True,
    help="Retrieval mode.",
)
@click.option(
    "--k",
    default=12,
    show_default=True,
    help="Number of chunks to include in the LLM context.",
)
@click.option(
    "--candidates",
    default=50,
    show_default=True,
    help="FTS candidate pool size for hybrid mode.",
)
@click.option(
    "--save-to",
    "save_to",
    default=None,
    type=click.Path(),
    help="Optional path to save the answer as a Markdown file.",
)
def ask(
    question: str,
    config_path: str | None,
    index_path_override: str | None,
    mode: str,
    k: int,
    candidates: int,
    save_to: str | None,
) -> None:
    """Answer QUESTION using retrieved vault chunks.

    QUESTION is a natural-language query.  The answer includes a Sources
    section listing the vault files that contributed to the response.
    """
    cfg = load_config(Path(config_path) if config_path else None)
    idx_path = _resolve_index_path(cfg, index_path_override)

    if not idx_path.exists():
        console.print(
            f"[red]Error:[/red] Index not found at {idx_path}.\n"
            "Run [bold]obsassist index build[/bold] first."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Retrieve chunks
    # ------------------------------------------------------------------
    chunks: list[ChunkResult] = []

    if mode == "fts":
        chunks = retrieve_fts(idx_path, question, k=k)

    elif mode in ("vector", "hybrid"):
        # Need embedding model to embed the query
        emb_client = OllamaClient(
            base_url=cfg.embeddings.base_url,
            model=cfg.embeddings.model,
        )
        if not emb_client.health_check():
            console.print(
                f"[red]Error:[/red] Cannot reach Ollama at "
                f"[bold]{cfg.embeddings.base_url}[/bold] for embeddings.\n"
                "Make sure Ollama is running: [bold]ollama serve[/bold]\n"
                "Then pull the embedding model: "
                f"[bold]ollama pull {cfg.embeddings.model}[/bold]"
            )
            sys.exit(1)

        try:
            with console.status("[yellow]Embedding query…[/yellow]"):
                query_vec = emb_client.embed(question)
        except Exception as exc:
            console.print(
                f"[red]Error:[/red] Failed to embed query: {exc}\n"
                f"Make sure the model [bold]{cfg.embeddings.model}[/bold] is available.\n"
                f"Run: [bold]ollama pull {cfg.embeddings.model}[/bold]"
            )
            sys.exit(1)

        if mode == "vector":
            chunks = retrieve_vector(idx_path, query_vec, k=k)
        else:
            chunks = retrieve_hybrid(idx_path, question, query_vec, k=k, candidates=candidates)

    if not chunks:
        console.print(
            "[yellow]No relevant chunks found.[/yellow] "
            "Try running [bold]obsassist index build[/bold] and/or "
            "[bold]obsassist embeddings build[/bold] first."
        )
        sys.exit(0)

    # ------------------------------------------------------------------
    # Generate answer
    # ------------------------------------------------------------------
    llm_client = _build_client(cfg)
    if not llm_client.health_check():
        console.print(
            f"[red]Error:[/red] Cannot reach Ollama at [bold]{cfg.ollama.base_url}[/bold].\n"
            "Make sure Ollama is running: [bold]ollama serve[/bold]"
        )
        sys.exit(1)

    context = _build_context(chunks)
    prompt = build_ask_prompt(context, question)

    console.print(
        f"[cyan]Asking:[/cyan] {question[:80]}{'…' if len(question) > 80 else ''}  "
        f"[dim]mode={mode}  chunks={len(chunks)}  model={cfg.ollama.model}[/dim]"
    )

    with console.status("[yellow]Calling Ollama…[/yellow]"):
        answer = llm_client.generate(prompt)

    sources_md = _format_sources(chunks)
    full_output = f"{answer.strip()}\n\n---\n\n**Sources**\n\n{sources_md}"

    console.print()
    console.print(Markdown(full_output))

    if save_to:
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        header = f"# {question}\n\n"
        save_path.write_text(header + full_output, encoding="utf-8")
        console.print(f"\n[green]✓ Saved to:[/green] {save_path}")


