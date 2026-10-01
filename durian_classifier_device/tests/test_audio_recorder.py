#!/usr/bin/env python3
"""Kiểm tra định dạng WAV bằng chế độ microphone mô phỏng."""

from __future__ import annotations

import sys
import tempfile
import wave
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from hardware.audio_recorder import AudioRecorder  # noqa: E402


def main() -> None:
    """Tạo WAV mô phỏng và xác nhận mono, 44.1 kHz, 16-bit, 5 giây."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "record.wav"
        recorder = AudioRecorder(output_path=output_path, mock=True)
        assert recorder.record_audio() == output_path
        assert recorder.wait_until_recording_started(timeout=0.1)

        with wave.open(str(output_path), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == 44_100
            assert wav_file.getnframes() == 220_500

    print("Audio recorder mock test: PASS")


if __name__ == "__main__":
    main()
