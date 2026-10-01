"""Structured JSON logging for pipeline stage boundaries (AGENTS.md rule 5)."""
import json
import logging
import time

_ROOT = "anveshak"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "stage": record.name.removeprefix(_ROOT + "."),
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str)


def get_logger(stage: str) -> logging.Logger:
    return logging.getLogger(f"{_ROOT}.{stage}")


def boundary(logger: logging.Logger, level: int, event: str, **fields) -> None:
    """Log one stage-boundary crossing. Cheap no-op when the level is disabled."""
    if logger.isEnabledFor(level):
        logger.log(level, event, extra={"fields": fields})


def configure(level: int = logging.INFO, path: str | None = None) -> None:
    handler = logging.FileHandler(path) if path else logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger(_ROOT)
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False
