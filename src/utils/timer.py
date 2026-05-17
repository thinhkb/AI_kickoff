"""
Measures inference time for time_response.
"""
import time
from contextlib import contextmanager


class Timer:
    """Simple timer for measuring inference latency."""

    def __init__(self):
        self._start = None
        self._elapsed = 0.0

    def start(self):
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is not None:
            self._elapsed = time.perf_counter() - self._start
            self._start = None
        return self._elapsed

    @property
    def elapsed(self) -> float:
        if self._start is not None:
            return time.perf_counter() - self._start
        return self._elapsed


@contextmanager
def timed():
    """Context manager that yields a Timer."""
    t = Timer()
    t.start()
    yield t
    t.stop()
