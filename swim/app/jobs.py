"""Background processing for the app's Videos page (docs/app.md): videos queued to run through run_pipeline(), one at
a time, with progress and cancel.

Results go to `<output>/<video name>/` as with `python -m swim.run --input <video>`. A cancelled or failed run
leaves the saved results consistent (run_pipeline() saves each stage only when it is done).
"""

import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from swim.app.library import stored_path
from swim.pipeline.config import PipelineConfig
from swim.pipeline.runner import run_pipeline

ACTIVE = ("queued", "running")


class Cancelled(Exception):
    """Raised in the progress callback to stop a run."""


@dataclass
class Job:
    id: int
    video: Path  # absolute
    name: str  # results folder name: <output>/<name>/
    config: PipelineConfig
    start: str  # first stage to run
    state: str = "queued"  # queued, running, done, failed, cancelled
    stage: str | None = None  # stage running
    done: int = 0  # frames done in that stage
    total: int = 0  # frames in the video
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    cancel_requested: bool = False
    rate_start: tuple[float, int] | None = None  # (time, frames done) at the first frame of the stage: speed so far

    def to_dict(self, now: float) -> dict:
        seconds_left = None
        if self.state == "running" and self.rate_start and self.done - self.rate_start[1] >= 5:
            t0, d0 = self.rate_start
            seconds_left = round((self.total - self.done) * (now - t0) / (self.done - d0), 1)
        end = self.finished or now
        return {
            "id": self.id, "video": str(self.video), "name": self.name, "start": self.start, "state": self.state,
            "stage": self.stage, "done": self.done, "total": self.total, "seconds_left": seconds_left,
            "seconds": round(end - self.started, 1) if self.started else None, "error": self.error,
            "cancel_requested": self.cancel_requested,
        }


class JobQueue:
    """Videos waiting to be processed; a background thread runs them one at a time."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._jobs: list[Job] = []
        self._changed = threading.Condition()  # guards _jobs and their fields
        self._worker: threading.Thread | None = None

    def add(self, video: Path, config: PipelineConfig, start: str) -> Job:
        """Queue a video. Raises ValueError if its results folder already has a queued or running job."""
        with self._changed:
            if self._active(video.stem):
                raise ValueError(f"'{video.stem}' is already being processed or waiting")
            job = Job(id=len(self._jobs) + 1, video=video, name=video.stem, config=config, start=start)
            self._jobs.append(job)
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, name="swim-jobs", daemon=True)
                self._worker.start()
            self._changed.notify_all()
            return job

    def cancel(self, job_id: int) -> bool:
        """Cancel a queued job, or stop a running one at its next frame. False if there is no such job."""
        with self._changed:
            job = next((j for j in self._jobs if j.id == job_id), None)
            if job is None:
                return False
            if job.state == "queued":
                job.state, job.finished = "cancelled", time.time()
            elif job.state == "running":
                job.cancel_requested = True
            self._changed.notify_all()
            return True

    def busy(self, name: str) -> bool:
        """Whether the results folder `name` has a queued or running job."""
        with self._changed:
            return self._active(name) is not None

    def jobs(self) -> list[dict]:
        """Every job of this session, oldest first; queued ones with their place in the queue (1 = next)."""
        now = time.time()
        with self._changed:
            out, waiting = [], 0
            for job in self._jobs:
                item = job.to_dict(now)
                if job.state == "queued":
                    waiting += 1
                    item["position"] = waiting
                out.append(item)
            return out

    def wait(self, timeout: float = 30) -> bool:
        """Wait until no job is queued or running (for tests). False on timeout."""
        deadline = time.time() + timeout
        with self._changed:
            while any(j.state in ACTIVE for j in self._jobs):
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                self._changed.wait(timeout=remaining)
            return True

    def _active(self, name: str) -> Job | None:
        return next((j for j in self._jobs if j.name == name and j.state in ACTIVE), None)

    def _work(self) -> None:
        while True:
            with self._changed:
                while (job := next((j for j in self._jobs if j.state == "queued"), None)) is None:
                    self._changed.wait()
                job.state, job.started = "running", time.time()
            self._run(job)

    def _run(self, job: Job) -> None:
        def progress(stage: str, done: int, total: int) -> None:
            with self._changed:
                if job.cancel_requested:
                    raise Cancelled
                if stage != job.stage:
                    job.stage, job.rate_start = stage, None
                if job.rate_start is None and done > 0:
                    job.rate_start = (time.time(), done)
                job.done, job.total = done, total

        state, error = "done", None
        try:
            run_pipeline(stored_path(job.video), self.output_dir / job.name, job.config, start=job.start,
                         progress=progress)
        except Cancelled:
            state = "cancelled"
        except Exception as e:  # shown on the page; the full trace goes to the server's console
            traceback.print_exc()
            state, error = "failed", f"{type(e).__name__}: {e}"
        with self._changed:
            job.state, job.error, job.finished = state, error, time.time()
            self._changed.notify_all()
