"""Simple in-memory cache with size limit."""

from __future__ import annotations

from typing import Any, Dict, Tuple


class SimpleCache:
    def __init__(self, max_items: int = 256) -> None:
        self._store: Dict[str, Tuple[Any, int]] = {}
        self._seq = 0
        self._max = max_items

    def get(self, key: str) -> Any:
        if key in self._store:
            val, _ = self._store[key]
            self._seq += 1
            self._store[key] = (val, self._seq)
            return val
        return None

    def set(self, key: str, value: Any) -> None:
        self._seq += 1
        self._store[key] = (value, self._seq)
        if len(self._store) > self._max:
            # Evict least recently used by seq
            oldest_key = min(self._store.items(), key=lambda kv: kv[1][1])[0]
            self._store.pop(oldest_key, None)


__all__ = ["SimpleCache"]



