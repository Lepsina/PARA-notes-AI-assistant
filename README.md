# PARA Notes AI Assistant

A Python CLI tool (`obsassist`) that analyzes your [Obsidian](https://obsidian.md) Markdown notes
using a locally-running [Ollama](https://ollama.com) model and inserts a structured
**`## Assistant`** block containing:

~~~markdown
## Assistant

### Summary
…

### Questions
- …

### Metadata suggestions
```yaml
…
```
~~~

The block is inserted at the end of the note on the first run and updated in-place on every
subsequent run — no other content is touched.  Before writing, the tool shows a colour diff and
asks for confirmation (pass `--yes` to skip).

It also provides a **full-text search index** powered by SQLite FTS5, **semantic embeddings**
powered by Ollama (`nomic-embed-text`), and an **`ask`** command that answers natural-language
questions about your vault using hybrid retrieval (FTS + vector reranking).

---

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.11 |
| [Ollama](https://ollama.com/download) | any recent version |
| LLM model | `llama3:8b` (or change in config) |
| Embedding model | `nomic-embed-text` (for embeddings / ask) |

> The indexing and search commands require **no additional dependencies** beyond the base install —
> SQLite FTS5 is bundled with Python's standard library.

---

## Installation (Windows)

```powershell
# 1. Clone the repository
git clone https://github.com/Lepsina/PARA-notes-AI-assistant.git
cd PARA-notes-AI-assistant

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install the package (editable mode keeps code changes live)
pip install -e .

# 4. Copy the sample config into your vault
copy config.example.yml "C:\Users\YOUR_USERNAME\Documents\Obsidian Vault\Assistant\config.yml"
# Edit the copy to set vault_root and adjust any settings.

# 5. Pull Ollama models
ollama pull llama3:8b          # for analyze / metadata / ask (generation)
ollama pull nomic-embed-text   # for embeddings / ask (retrieval)
```

After installation the `obsassist` command is available in your activated virtual environment.

---

## Configuration

Copy `config.example.yml` to your vault under `Assistant/config.yml` and edit as needed:

```yaml
vault_root: "C:/Users/YOUR_USERNAME/Documents/Obsidian Vault"

exclude_paths:
  - Resources/
  - Templates/
  - Files/
  - Excalidraw/
  - .obsidian/

# Optional: override the default index location
# index_path: "C:/Users/YOUR_USERNAME/AppData/Local/obsassist/index.sqlite"

ollama:
  base_url: "http://127.0.0.1:11434"
  model: "llama3:8b"
  temperature: 0.2

embeddings:
  base_url: "http://127.0.0.1:11434"
  model: "nomic-embed-text"
  chunk_size: 1000      # max characters per chunk
  chunk_overlap: 200    # overlap between adjacent chunks
  batch_size: 32
```

All fields are optional — the tool falls back to sensible defaults.

---

### Metadata vocabulary file

Create `Assistant/metadata.vocab.yml` in your vault (see the bundled example file) and
reference it from `config.yml`:

```yaml
metadata:
  vocab_path: "C:/Users/YOUR_USERNAME/Documents/Obsidian Vault/Assistant/metadata.vocab.yml"
```

The vocabulary file controls:

* `topics_allowed` — canonical topic identifiers that are kept in the `topics` field.
* `normalize_topics` — map from raw tags/topic strings to canonical topics
  (e.g. `матан → calculus`, `bfs → graph-algorithms`).
* `priority_from_tags` — extract `priority` from existing tags
  (e.g. `#High → priority: high`).
* `workflow_from_tags` — optional workflow label mapping (informational).

A copy of the vocabulary file with the user's full tag set is bundled at
`Assistant/metadata.vocab.yml`.

---

### Separate model for metadata

You can use a smaller, faster model for the `metadata` command while keeping a larger model
for `analyze` / `ask`:

```yaml
ollama:
  model: "llama3:8b"      # used by analyze and ask

metadata:
  model: "qwen2.5:3b-instruct"  # used by metadata (falls back to ollama.model if blank)
```

Pull the metadata model once:

```powershell
ollama pull qwen2.5:3b-instruct
```

If `metadata.model` is not set or the model is unavailable, the command automatically
falls back to `ollama.model`.


## Usage

Make sure Ollama is running first:

```powershell
ollama serve           # in a separate terminal
ollama pull llama3:8b  # one-time download
```

### Analyze a note (Summary + Questions + Metadata)

```powershell
obsassist analyze --file "C:\Users\YOUR_USERNAME\Documents\Obsidian Vault\Areas\self-expression.md"
```

### Update metadata in YAML frontmatter (strict mode)

The `metadata` command updates **only** the YAML frontmatter block at the top of the note.
The markdown body is **never modified** — it remains byte-for-byte identical after the command.

```powershell
obsassist metadata --file "C:\Users\YOUR_USERNAME\Documents\Obsidian Vault\Areas\self-expression.md"
```

#### What it does

1. Calls a lightweight LLM to suggest frontmatter fields.
2. Validates the output against a strict **schema allowlist** (unknown keys are silently
   dropped).
3. Applies **conservative merge**: only missing / empty fields are filled; existing
   user-authored values are preserved.
4. Normalises values using an optional external **vocabulary file**
   (`topics`, `priority`, status aliases).
5. Shows a colour diff and asks for confirmation before writing.

#### `--force` flag

```powershell
obsassist metadata --file note.md --force
```

Overwrites all existing frontmatter fields with the LLM suggestions (full regeneration).

#### Allowed frontmatter keys

| Key | Type | Notes |
|---|---|---|
| `title` | string | Note title |
| `created` | string | Creation date (`YYYY-MM-DD`) |
| `updated` | string | Last update date (auto-set on change) |
| `type` | string | Note type |
| `status` | string | `draft` \| `active` \| `done` |
| `lang` | string | Language code |
| `tags` | list | Original tags preserved |
| `topics` | list | Canonical topics (normalised via vocab) |
| `entities` | list | Named entities |
| `summary` | string | One-line description |
| `priority` | string | `high` \| `medium` \| `low` |
| `source_type` | string | Source type |
| `exclude_from_ai` | boolean | Skip in AI queries |
| `aliases` | list | Note aliases |

Keys **not** in this list are discarded unless added via `extra_allowed_keys` in config.

#### Legacy aliases (automatically normalised)

| Input key / value | Canonical output |
|---|---|
| `topic:` | `topics:` (list) |
| `tag:` | `tags:` (list) |
| `status: complete` | `status: done` |
| `status: completed` | `status: done` |
| `status: in_progress` | `status: active` |
| `status: in-progress` | `status: active` |


### Skip confirmation prompt

```powershell
obsassist analyze --file note.md --yes
```

### Use a custom config file

```powershell
obsassist analyze --file note.md --config path\to\config.yml
```

---

## Indexing

The index lets you search your entire vault for notes that contain specific words or phrases.
It is stored **outside the vault** so sync tools (iCloud, OneDrive, Obsidian Sync) never touch it.

### Default index location

| Platform | Path |
|---|---|
| Windows | `%LOCALAPPDATA%\obsassist\index.sqlite` (e.g. `C:\Users\lepsina\AppData\Local\obsassist\index.sqlite`) |
| Linux / macOS | `~/.local/share/obsassist/index.sqlite` |

You can override the location via `index_path` in `config.yml` or with the `--index-path` flag.
Parent directories are created automatically.

### Build the index (full rebuild)

Run this once after installation, or any time you want a fresh start:

```powershell
obsassist index build --config "C:\path\to\config.yml"
```

### Update the index (incremental)

Run this regularly to pick up new and changed notes without re-indexing everything:

```powershell
obsassist index update --config "C:\path\to\config.yml"
```

Only files whose `mtime`, `size`, or content hash have changed are re-indexed.
Deleted files are automatically removed from the index.

### Search the index

```powershell
obsassist search "тщеславие" --config "C:\path\to\config.yml"   # Cyrillic (Unicode is fully supported)
obsassist search "creative writing" --limit 10
```

The search uses SQLite FTS5 — plain-text queries work out of the box, and FTS5 query syntax
(phrase search `"two words"`, negation `-word`, column filters `title:word`) is also supported.

Output is a ranked table with file path, note title, and a content snippet.

### Override the index path on the command line

```powershell
obsassist index build --index-path D:\my-index.sqlite
obsassist search "query"  --index-path D:\my-index.sqlite
```

---

## Semantic Embeddings

The embeddings index stores text chunks and their vector representations, enabling semantic
(meaning-based) search that goes beyond keyword matching.

### Setup

```powershell
ollama pull nomic-embed-text   # one-time download (~270 MB)
```

### Build the embeddings index (full rebuild)

```powershell
obsassist embeddings build --config "C:\path\to\config.yml"
```

This chunks all vault notes and embeds them with `nomic-embed-text`.

### Update the embeddings index (incremental)

```powershell
obsassist embeddings update --config "C:\path\to\config.yml"
```

Only files whose content has changed are re-chunked and re-embedded.

---

## Ask Command

Answer natural-language questions about your vault with cited sources.

```powershell
obsassist ask "Какие у меня цели на этот квартал?" --config "C:\path\to\config.yml"
obsassist ask "What projects am I working on?" --mode fts
obsassist ask "Tell me about my fitness notes" --mode vector --k 8
```

### Retrieval modes

| Mode | Description |
|---|---|
| `hybrid` (default) | FTS gathers candidates, then reranks by vector similarity |
| `fts` | Full-text search only (no embedding model required) |
| `vector` | Pure cosine similarity over all chunk embeddings |

### Options

| Flag | Default | Description |
|---|---|---|
| `--mode` | `hybrid` | Retrieval strategy |
| `--k` | `12` | Number of chunks to include in the LLM context |
| `--candidates` | `50` | FTS candidate pool size for hybrid mode |
| `--save-to` | — | Save the answer as a Markdown file |

### Example output

```
Analyzing: "Какие у меня цели на этот квартал?"  mode=hybrid  chunks=8  model=llama3:8b

В этом квартале ваши основные цели…

---

**Sources**

- Projects/Q2-goals.md — Goals Overview
- Daily/2026-04-01.md — Weekly review
```

### Prerequisites for ask

- `fts` mode: run `obsassist index build` first.
- `vector` / `hybrid` mode: run both `obsassist index build` **and** `obsassist embeddings build`.

---

## Obsidian integration (Shell Commands plugin)

Install the [Shell Commands](https://github.com/Taitava/obsidian-shellcommands) community plugin,
then add a new command:

| Field | Value |
|---|---|
| Shell command | `obsassist analyze --file "{{file_path:absolute}}" --config "C:/Users/YOUR_USERNAME/Documents/Obsidian Vault/Assistant/config.yml"` |
| Alias | `AI: Analyze note` |
| Shell | PowerShell |

Assign a hotkey (e.g. `Ctrl+Shift+A`) to trigger the analysis from inside Obsidian.

> **Tip:** use `--yes` in the shell command to apply changes automatically, or omit it to review
> the diff in the terminal before confirming.

---

## Troubleshooting

### Ollama not reachable on Windows / PowerShell

If `obsassist` reports "Cannot reach Ollama" but `ollama serve` is running:

```powershell
# 1. Clear proxy environment variables that may interfere
Remove-Item Env:HTTP_PROXY  -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:ALL_PROXY   -ErrorAction SilentlyContinue
$env:NO_PROXY = "127.0.0.1,localhost"

# 2. Force 127.0.0.1 (avoids IPv6 localhost resolution)
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# 3. Verify Ollama is actually responding
Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags"

# 4. Re-run your command
obsassist analyze --file note.md --config config.yml
```

The config default is already `http://127.0.0.1:11434`.  If you previously had
`http://localhost:11434` in your `config.yml`, update it to `127.0.0.1`.

### Embedding model not available

```powershell
ollama pull nomic-embed-text
```

If `embeddings build` is slow, reduce `batch_size` in config (default: 32).

---

## Project structure

```
PARA-notes-AI-assistant/
├── pyproject.toml          # build config, dependencies, obsassist entrypoint
├── config.example.yml      # sample configuration (copy to vault)
├── Assistant/
│   └── metadata.vocab.yml  # vocabulary file for tag/topic normalisation
├── obsassist/
│   ├── cli.py              # Click commands: analyze, metadata, index, search, embeddings, ask
│   ├── config.py           # YAML config loading + MetadataConfig + EmbeddingsConfig + get_index_path()
│   ├── metadata_guard.py   # Strict frontmatter guardrails: sanitize, merge, vocab, write
│   ├── indexer.py          # SQLite FTS5 index build/update + metadata extraction
│   ├── search.py           # Full-text search over the FTS5 index
│   ├── chunker.py          # Markdown-aware text chunker (heading + size split)
│   ├── embeddings.py       # Embeddings pipeline (chunks + vectors tables)
│   ├── retrieval.py        # FTS / vector / hybrid retrieval strategies
│   ├── parser.py           # ## Assistant block insert/update/parse
│   ├── filters.py          # exclude-path logic
│   ├── diff.py             # unified diff generation
│   ├── ollama_client.py    # HTTP wrapper for Ollama /api/generate + /api/embeddings
│   └── prompts.py          # prompt templates + response parser
└── tests/
    ├── test_metadata_guard.py  # strict metadata guardrails (56 tests)
    ├── test_parser.py
    ├── test_filters.py
    ├── test_diff.py
    ├── test_indexer.py
    ├── test_search.py
    ├── test_chunker.py
    ├── test_embeddings.py
    ├── test_retrieval.py
    └── test_ask_cmd.py
```

---

## Development

```powershell
pip install -e ".[dev]"
python -m pytest
```

---

## Excluded paths

The following vault directories are **never** read or modified by any command:

| Path | Reason |
|---|---|
| `Resources/` | Reference material, not personal notes |
| `Templates/` | Template files |
| `Files/` | Attachments |
| `Excalidraw/` | Diagram files |
| `.obsidian/` | Obsidian internal config |

`Archive/` is **allowed** but has a lower priority weighting for future batch processing.

