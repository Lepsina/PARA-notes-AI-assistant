"""Configuration loading for obsassist."""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3:8b"
    temperature: float = 0.2


@dataclass
class AssistantBlockConfig:
    heading: str = "## Assistant"
    sections: list[str] = field(
        default_factory=lambda: ["Summary", "Questions", "Metadata suggestions"]
    )


@dataclass
class Config:
    vault_root: str = ""  # Set this in your config.yml (see config.example.yml)
    include_extensions: list[str] = field(default_factory=lambda: [".md"])
    exclude_paths: list[str] = field(
        default_factory=lambda: [
            "Resources/",
            "Templates/",
            "Files/",
            "Excalidraw/",
            ".obsidian/",
        ]
    )
    path_priority: dict[str, float] = field(
        default_factory=lambda: {
            "Daily": 1.0,
            "Buffer": 0.9,
            "Projects": 0.9,
            "Areas": 0.8,
            "Archive": 0.2,
        }
    )
    date_format: str = "%Y-%m-%d"
    # Leave empty to use the platform default (see get_index_path()).
    index_path: str = ""
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    assistant_block: AssistantBlockConfig = field(default_factory=AssistantBlockConfig)


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from a YAML file, falling back to defaults.

    Args:
        config_path: Explicit path to a config YAML file. When *None* the
            function searches for ``config.yml`` / ``config.yaml`` in the
            current working directory before returning a default Config.
    """
    if config_path is None:
        for candidate in (Path("config.yml"), Path("config.yaml")):
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None:
        return Config()

    with open(config_path, encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    return _parse_config(data)


def _parse_config(data: dict[str, Any]) -> Config:
    cfg = Config()

    if "vault_root" in data:
        cfg.vault_root = str(data["vault_root"])
    if "include_extensions" in data:
        cfg.include_extensions = list(data["include_extensions"])
    if "exclude_paths" in data:
        cfg.exclude_paths = list(data["exclude_paths"])
    if "path_priority" in data:
        cfg.path_priority = {k: float(v) for k, v in data["path_priority"].items()}
    if "date_format" in data:
        cfg.date_format = str(data["date_format"])
    if "index_path" in data:
        cfg.index_path = str(data["index_path"])

    if "ollama" in data:
        od = data["ollama"]
        cfg.ollama = OllamaConfig(
            base_url=str(od.get("base_url", "http://localhost:11434")),
            model=str(od.get("model", "llama3:8b")),
            temperature=float(od.get("temperature", 0.2)),
        )

    if "assistant_block" in data:
        abd = data["assistant_block"]
        cfg.assistant_block = AssistantBlockConfig(
            heading=str(abd.get("heading", "## Assistant")),
            sections=list(
                abd.get("sections", ["Summary", "Questions", "Metadata suggestions"])
            ),
        )

    return cfg


def get_index_path(cfg: Config) -> Path:
    """Return the resolved path to the SQLite index file.

    Priority:
    1. ``cfg.index_path`` if explicitly set (config file or CLI flag).
    2. ``%LOCALAPPDATA%\\obsassist\\index.sqlite`` on Windows.
    3. ``$XDG_DATA_HOME/obsassist/index.sqlite`` (or
       ``~/.local/share/obsassist/index.sqlite``) on Linux/macOS.
    """
    if cfg.index_path:
        return Path(cfg.index_path)

    if platform.system() == "Windows":
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            base = Path(local_app_data)
        else:
            base = Path.home() / "AppData" / "Local"
    else:
        xdg = os.environ.get("XDG_DATA_HOME", "")
        base = Path(xdg) if xdg else Path.home() / ".local" / "share"

    return base / "obsassist" / "index.sqlite"
