
import threading
import time
from collections.abc import Callable

DEFAULT_CAPACITY = 80
DEFAULT_PERIOD = 240.0
DEFAULT_MAX_CLIENTS = 4096

class TokenBucket:

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        period: float = DEFAULT_PERIOD,
        max_clients: int = DEFAULT_MAX_CLIENTS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        self._capacity = float(capacity)
        self._rate = capacity / period
        self._max_clients = max_clients
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, float]] = {}

    def check(self, key: str) -> float | None:
        now = self._clock()
        with self._lock:
            tokens, seen = self._buckets.get(key, (self._capacity, now))
            tokens = min(self._capacity, tokens + (now - seen) * self._rate)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return (1.0 - tokens) / self._rate
            self._buckets[key] = (tokens - 1.0, now)
            self._prune_locked(now)
            return None

    def _prune_locked(self, now: float) -> None:
        if len(self._buckets) <= self._max_clients:
            return
        full_after = self._capacity / self._rate
        self._buckets = {
            key: entry
            for key, entry in self._buckets.items()
            if now - entry[1] < full_after
        }

    def __len__(self) -> int:
        with self._lock:
            return len(self._buckets)
