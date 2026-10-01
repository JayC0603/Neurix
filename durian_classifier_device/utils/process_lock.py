"""Process-wide lock used to prevent two hardware owners running at once."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import TextIO


class AlreadyRunningError(RuntimeError):
    """Raised when another application process already owns the lock."""


class ProcessLock:
    """Hold an advisory file lock for the lifetime of the application."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: TextIO | None = None

    def acquire(self) -> None:
        if self._file is not None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.seek(0)
            owner = lock_file.read().strip()
            detail = f" (PID {owner})" if owner else ""
            lock_file.close()
            raise AlreadyRunningError(
                "Durian classifier is already running"
                f"{detail}. If it is managed by systemd, use "
                "'sudo systemctl restart durian-classifier.service' instead "
                "of starting scripts/run.sh again."
            ) from exc

        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        self._file = lock_file

    def release(self) -> None:
        if self._file is None:
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

