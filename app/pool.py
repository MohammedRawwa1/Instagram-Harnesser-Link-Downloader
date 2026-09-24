"""Adaptive worker pool.

Maintains a concurrency limit that rises and falls based on live signals
instead of a fixed number. Designed to "smell" rate limiting before it
cascades into hard errors.

Signals:
  * ``mark_rate_limited()`` — a 429/Too Many Requests just happened.
    Shrinks the pool immediately and opens a cooldown window.
  * ``mark_slow(ttf_ms)`` — a request just completed, record its time-to-first
    byte / round-trip. If recent latency is climbing, the pool preemptively
    reduces concurrency.
  * ``try_acquire()`` — async context manager to enter the pool. Blocks (with
    a timeout so tasks can bail instead of hanging forever) until a slot is
    available, then gives you a token to ``release()``.

Ramping policy:
  * After a rate-limit event, the pool stays low for ``cooldown`` seconds,
    then ramps back up gradually (one extra slot every ``ramp_interval`` s)
    until it hits ``max_workers``.
  * If no errors for ``stable_window`` seconds, the pool can grow back to
    ``max_workers``.
  * ``min_workers`` is the floor — we never go to zero because that would
    stall everything.

Usage:
    pool = WorkerPool(max_workers=8, min_workers=1)
    async with pool.acquire():
        result = await do_work()
    pool.mark_slow(ttf_ms=120)
    # or, on 429:
    pool.mark_rate_limited()
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Optional


class WorkerPool:
    """An adaptive concurrency limiter with predictive back-off."""

    def __init__(
        self,
        *,
        max_workers: int = 8,
        min_workers: int = 1,
        cooldown: float = 30.0,
        ramp_interval: float = 5.0,
        stable_window: float = 60.0,
        slow_threshold_ms: float = 2000.0,
        slow_scale_factor: float = 0.5,
        max_history: int = 50,
    ):
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if min_workers < 1 or min_workers > max_workers:
            raise ValueError(f"min_workers must be in [1, {max_workers}]")

        self.max_workers = max_workers
        self.min_workers = min_workers
        self.cooldown = cooldown
        self.ramp_interval = ramp_interval
        self.stable_window = stable_window
        self.slow_threshold_ms = slow_threshold_ms
        self.slow_scale_factor = slow_scale_factor

        self._limit = max_workers
        self._lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(self._limit)

        # Recent round-trip times (seconds).
        self._rtts: deque[float] = deque(maxlen=max_history)
        # Timestamps of recent rate-limit events.
        self._rl_events: deque[float] = deque(maxlen=max_history)

        # When did we last see a rate-limit event? Used for cooldown.
        self._last_rl_at: float = 0.0
        # Last time the pool adjusted (for ramp scheduling).
        self._last_adjust_at: float = 0.0

        # Background re-evaluator.
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # ------------------------------------------------------------------
    # Public signals
    # ------------------------------------------------------------------

    def mark_rate_limited(self) -> None:
        """A 429 / rate-limit just happened. Shrink the pool."""
        now = time.monotonic()
        self._rl_events.append(now)
        self._last_rl_at = now
        self._decay(limit=max(self.min_workers, self._limit - 2))

    def mark_slow(self, ttf_ms: float) -> None:
        """Record a request's round-trip time; possibly shrink if latency
        is climbing."""
        self._rtts.append(ttf_ms / 1000.0)
        if ttf_ms > self.slow_threshold_ms:
            # Predictive: latency is high → reduce before more 429s pile up.
            self._decay(limit=max(self.min_workers, int(self._limit * self.slow_scale_factor)))

    def mark_fast(self, ttf_ms: float) -> None:
        """Record a fast request; used for the recovery path."""
        self._rtts.append(ttf_ms / 1000.0)

    # ------------------------------------------------------------------
    # Acquire / release
    # ------------------------------------------------------------------

    async def acquire(self, timeout: float = 30.0) -> "_Token":
        """Wait until a slot is available (or timeout). Returns a token that
        must be released via ``async with`` or ``token.release()``."""
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"worker pool busy (limit={self._limit}); "
                f"no slot in {timeout}s"
            )
        return _Token(self)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _decay(self, limit: int) -> None:
        """Lower the concurrency limit (called under lock by signal methods)."""
        if limit < self._limit:
            self._limit = limit
            self._sem = asyncio.Semaphore(limit)
            self._last_adjust_at = time.monotonic()

    async def _reevaluate(self) -> None:
        """Background loop: slowly ramp back up when things are healthy."""
        while self._running:
            await asyncio.sleep(1.0)
            async with self._lock:
                now = time.monotonic()
                since_rl = now - self._last_rl_at

                # Cooldown after a rate-limit event: stay low.
                if since_rl < self.cooldown:
                    continue

                # If we're far below max and the coast is clear, ramp up.
                if self._limit < self.max_workers:
                    if since_rl >= self.cooldown and len(self._rl_events) <= 1:
                        # No recent rate limiting — ramp toward max.
                        elapsed_since_adjust = now - self._last_adjust_at
                        if elapsed_since_adjust >= self.ramp_interval:
                            new_limit = min(self.max_workers, self._limit + 1)
                            if new_limit > self._limit:
                                self._limit = new_limit
                                self._sem = asyncio.Semaphore(new_limit)
                                self._last_adjust_at = now

                # Stable window: if no errors for a while, allow full throttle.
                if (
                    since_rl >= self.stable_window
                    and self._limit < self.max_workers
                    and not self._rl_events
                ):
                    self._limit = self.max_workers
                    self._sem = asyncio.Semaphore(self.max_workers)
                    self._last_adjust_at = now
                    self._rl_events.clear()

    def start(self) -> None:
        """Start the background re-evaluation loop. Call once per pool."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._reevaluate(), name="worker-pool-reeval")

    async def close(self) -> None:
        """Stop the background loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    @property
    def limit(self) -> int:
        """Current concurrency limit."""
        return self._limit


class _Token:
    """A leased slot in the pool. Release it when the work is done."""

    def __init__(self, pool: WorkerPool) -> None:
        self._pool = pool

    def release(self) -> None:
        self._pool._sem.release()

    async def __aenter__(self) -> _Token:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.release()
