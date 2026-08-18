"""
RateLimiter: sliding-window rate limiter for API calls.

Acts as the agent's "actionable energy" — each external API call
consumes a token. When tokens are exhausted, the agent must wait.
"""

import logging
import time

logger = logging.getLogger("RateLimiter")


class RateLimiter:
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: list[float] = []

    def acquire(self) -> float:
        """
        Wait if rate limit is exceeded. Returns the time waited (seconds).
        """
        now = time.time()
        self._clean(now)

        if len(self._timestamps) >= self.max_requests:
            oldest = self._timestamps[0]
            wait = self.window_seconds - (now - oldest)
            if wait > 0:
                logger.info(f"Rate limit reached. Waiting {wait:.1f}s...")
                time.sleep(wait)
                now = time.time()
                self._clean(now)

        self._timestamps.append(time.time())
        return 0.0

    def _clean(self, now: float):
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    @property
    def remaining(self) -> int:
        self._clean(time.time())
        return max(0, self.max_requests - len(self._timestamps))

    @property
    def reset_in(self) -> float:
        """Seconds until the rate limit window resets (0 = no wait needed)."""
        now = time.time()
        self._clean(now)
        if not self._timestamps or len(self._timestamps) < self.max_requests:
            return 0.0
        oldest = self._timestamps[0]
        wait = self.window_seconds - (now - oldest)
        return max(0.0, wait)

    @property
    def state(self) -> dict:
        return {
            "max": self.max_requests,
            "window": self.window_seconds,
            "used": len(self._timestamps),
            "remaining": self.remaining,
            "reset_in": round(self.reset_in, 1),
        }
