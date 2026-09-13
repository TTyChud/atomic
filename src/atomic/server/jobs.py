import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

DEFAULT_MAX_JOBS = 16

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"

_FINISHED = (JobStatus.DONE, JobStatus.ERROR)

@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    result: Any = None
    error: str | None = None

class JobStore:
    def __init__(
        self,
        max_jobs: int = DEFAULT_MAX_JOBS,
        on_evict: Callable[[str], None] | None = None,
    ) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

        self._on_evict = on_evict

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
            evicted = self._evict_locked()

        for job_id in evicted:
            if self._on_evict is not None:
                self._on_evict(job_id)
        return job

    def _evict_locked(self) -> list[str]:
        evicted: list[str] = []
        while len(self._jobs) > self._max_jobs:
            oldest = next(
                (jid for jid, job in self._jobs.items() if job.status in _FINISHED),
                None,
            )
            if oldest is None:
                break
            del self._jobs[oldest]
            evicted.append(oldest)
        return evicted

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    def run(self, job_id: str, fn: Callable[[Callable[[float], None]], Any]) -> None:
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"unknown job id: {job_id}")
        job.status = JobStatus.RUNNING

        def report(fraction: float) -> None:
            job.progress = min(max(fraction, 0.0), 1.0)

        try:
            job.result = fn(report)
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = JobStatus.ERROR
        else:
            job.progress = 1.0
            job.status = JobStatus.DONE
