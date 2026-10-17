from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def default_log_path() -> Path:
    log_dir = os.getenv("LOG_DIR")
    if log_dir:
        return Path(log_dir) / "requests.jsonl"
    return PROJECT_ROOT / "logs" / "requests.jsonl"


class RequestLogger:
    def __init__(self, log_path: str | Path | None = None) -> None:
        self.path = Path(log_path) if log_path else default_log_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, **fields: Any) -> None:
        record = {
            "ts": time.time(),
            "event_type": event_type,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
