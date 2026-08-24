"""Shared helper utilities for the context-engineering API.

Modules:
  - summarizer: LLM-backed conversation-compaction summaries
  - time_utils : timezone-safe datetime parsing / expiry & staleness checks
"""
from api.v1.utils.summarizer import (
    summarize_trimmed_turns,
    summarize_window,
    fold_blocks_into_meta,
)
from api.v1.utils.time_utils import (
    utc_now,
    parse_iso_utc,
    is_expired,
    is_stale,
)

__all__ = [
    "summarize_trimmed_turns",
    "summarize_window",
    "fold_blocks_into_meta",
    "utc_now",
    "parse_iso_utc",
    "is_expired",
    "is_stale",
]
