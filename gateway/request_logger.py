from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RequestLogger:
    def __init__(self, log_path: str | Path = "logs/requests.jsonl") -> None:
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, **fields: Any) -> None:
        record = {
            "ts": time.time(),
            "event_type": event_type,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")