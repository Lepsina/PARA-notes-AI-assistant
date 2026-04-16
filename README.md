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

---

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.11 |
| [Ollama](https://ollama.com/download) | any recent version |
| Model | `llama3:8b` (or change in config) |

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

ollama:
  base_url: "http://localhost:11434"
  model: "llama3:8b"
  temperature: 0.2
```

All fields are optional — the tool falls back to sensible defaults.

---

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
│   ├── cli.py              # Click commands: analyze, metadata
│   ├── config.py           # YAML config loading
│   ├── parser.py           # ## Assistant block insert/update/parse
│   ├── filters.py          # exclude-path logic
│   ├── diff.py             # unified diff generation
│   ├── ollama_client.py    # HTTP wrapper for Ollama /api/generate
│   └── prompts.py          # prompt templates + response parser
└── tests/
    ├── test_parser.py
    ├── test_filters.py
    └── test_diff.py
```

---

## Development

```powershell
pip install -e ".[dev]"
python -m pytest
```

---

## Excluded paths

The following vault directories are **never** read or modified:

| Path | Reason |
|---|---|
| `Resources/` | Reference material, not personal notes |
| `Templates/` | Template files |
| `Files/` | Attachments |
| `Excalidraw/` | Diagram files |
| `.obsidian/` | Obsidian internal config |

`Archive/` is **allowed** but has a lower priority weighting for future batch processing.
