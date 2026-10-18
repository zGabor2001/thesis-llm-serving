from __future__ import annotations

import os
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "models" / "base" / "tokenizer.json"

_cache: dict[Path, "PromptTokenizer"] = {}
_cache_lock = threading.Lock()


def fallback_token_count(text: str) -> int:
    """~4 chars/token heuristic, used when no real tokenizer is available."""
    return max(1, (len(text) + 3) // 4)


def default_tokenizer_path() -> Path:
    return Path(os.getenv("TOKENIZER_PATH", str(DEFAULT_TOKENIZER_PATH)))


class PromptTokenizer:
    """Token counter for DRR request costs.

    Loads a HuggingFace ``tokenizer.json`` (the fast-tokenizer format) when
    available; otherwise falls back to a characters-per-token heuristic so the
    gateway still runs without model files present (mock mode, containers
    without the tokenizer bind-mounted).
    """

    def __init__(self, tokenizer_path: str | Path | None = None) -> None:
        self.path = Path(tokenizer_path) if tokenizer_path is not None else default_tokenizer_path()
        self._tokenizer = None
        self._load_error: Exception | None = None
        self._lock = threading.Lock()

        if self.path.is_file():
            try:
                from tokenizers import Tokenizer

                self._tokenizer = Tokenizer.from_file(str(self.path))
            except Exception as exc:  # noqa: BLE001
                self._load_error = exc
        else:
            self._load_error = FileNotFoundError(self.path)

    @property
    def is_loaded(self) -> bool:
        return self._tokenizer is not None

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is not None:
            with self._lock:
                try:
                    return max(1, len(self._tokenizer.encode(text).ids))
                except Exception:  # noqa: BLE001
                    pass
        return fallback_token_count(text)


def get_tokenizer(tokenizer_path: str | Path | None = None) -> PromptTokenizer:
    """Return a cached PromptTokenizer for the given (or default) path."""
    path = Path(tokenizer_path) if tokenizer_path is not None else default_tokenizer_path()
    with _cache_lock:
        if path not in _cache:
            _cache[path] = PromptTokenizer(path)
        return _cache[path]
