"""Tests for single-instance hardware ownership."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.process_lock import AlreadyRunningError, ProcessLock


def test_second_process_lock_is_rejected(tmp_path: Path) -> None:
    lock_path = tmp_path / "app.lock"
    first = ProcessLock(lock_path)
    second = ProcessLock(lock_path)

    first.acquire()
    try:
        with pytest.raises(AlreadyRunningError, match="already running"):
            second.acquire()
    finally:
        first.release()


def test_process_lock_can_be_acquired_again_after_release(tmp_path: Path) -> None:
    lock_path = tmp_path / "app.lock"
    first = ProcessLock(lock_path)
    second = ProcessLock(lock_path)

    first.acquire()
    first.release()
    second.acquire()
    second.release()
