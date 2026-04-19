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

It also provides a **full-text search index** powered by SQLite FTS5 so you can search across your
entire vault from the command line without touching Obsidian.

---

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.11 |
| [Ollama](https://ollama.com/download) | any recent version (for analyze/metadata) |
| Model | `llama3:8b` (or change in config) |

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
  base_url: "http://localhost:11434"
  model: "llama3:8b"
  temperature: 0.2
```

All fields are optional — the tool falls back to sensible defaults.

---

## Usage

Make sure Ollama is running first (only needed for `analyze` / `metadata`):

```powershell
ollama serve           # in a separate terminal
ollama pull llama3:8b  # one-time download
```

### Analyze a note (Summary + Questions + Metadata)

```powershell
obsassist analyze --file "C:\Users\YOUR_USERNAME\Documents\Obsidian Vault\Areas\self-expression.md"
```

### Update metadata suggestions only

Existing `### Summary` and `### Questions` content is preserved.

```powershell
obsassist metadata --file "C:\Users\YOUR_USERNAME\Documents\Obsidian Vault\Areas\self-expression.md"
```

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

## Project structure

```
PARA-notes-AI-assistant/
├── pyproject.toml          # build config, dependencies, obsassist entrypoint
├── config.example.yml      # sample configuration (copy to vault)
├── obsassist/
│   ├── cli.py              # Click commands: analyze, metadata, index, search
│   ├── config.py           # YAML config loading + get_index_path()
│   ├── indexer.py          # SQLite FTS5 index build/update + metadata extraction
│   ├── search.py           # Full-text search over the FTS5 index
│   ├── parser.py           # ## Assistant block insert/update/parse
│   ├── filters.py          # exclude-path logic
│   ├── diff.py             # unified diff generation
│   ├── ollama_client.py    # HTTP wrapper for Ollama /api/generate
│   └── prompts.py          # prompt templates + response parser
└── tests/
    ├── test_parser.py
    ├── test_filters.py
    ├── test_diff.py
    ├── test_indexer.py
    └── test_search.py
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
