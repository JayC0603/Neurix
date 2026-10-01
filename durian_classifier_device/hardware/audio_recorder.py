"""Ghi âm mono 16-bit từ USB microphone và lưu thành tệp WAV."""

from __future__ import annotations

import logging
import subprocess
import threading
import wave
from pathlib import Path


LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 44_100
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
RECORD_SECONDS = 5.0
DEFAULT_ALSA_DEVICE = "plughw:CARD=Device,DEV=0"


class AudioRecorder:
    """Ghi một đoạn âm thanh tại một thời điểm mà không tạo worker riêng."""

    def __init__(
        self,
        output_path: Path,
        sample_rate: int = SAMPLE_RATE,
        duration_seconds: float = RECORD_SECONDS,
        alsa_device: str = DEFAULT_ALSA_DEVICE,
        mock: bool = False,
    ) -> None:
        self.output_path = Path(output_path)
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        # Dùng tên card USB ổn định thay cho số card có thể đổi sau mỗi lần khởi động.
        self.alsa_device = alsa_device
        self.mock = mock
        # Khóa ngăn hai lần ghi âm sử dụng USB microphone cùng lúc.
        self._recording_lock = threading.Lock()
        # Cho pipeline biết microphone đã bắt đầu thu trước khi servo phát tiếng gõ.
        self._capture_started = threading.Event()

    def wait_until_recording_started(self, timeout: float = 2.0) -> bool:
        """Chờ thiết bị thu âm bắt đầu nhận mẫu."""
        return self._capture_started.wait(timeout)

    def record_audio(self) -> Path:
        """Ghi mono 44.1 kHz, 16-bit đủ dài để thu trọn chuỗi gõ servo."""
        if not self._recording_lock.acquire(blocking=False):
            raise RuntimeError("Microphone is already recording")

        try:
            self._capture_started.clear()
            frame_count = int(self.sample_rate * self.duration_seconds)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.output_path.with_suffix(".tmp.wav")

            if self.mock:
                # Chế độ mô phỏng tạo đúng định dạng WAV nhưng không truy cập microphone.
                self._capture_started.set()
                audio_bytes = bytes(frame_count * CHANNELS * SAMPLE_WIDTH_BYTES)
                self._write_wave(temporary_path, audio_bytes)
            else:
                try:
                    import sounddevice as sd
                except (ImportError, OSError):
                    # Dùng arecord có sẵn nếu Python chưa nạp được PortAudio.
                    LOGGER.warning("sounddevice is unavailable; falling back to arecord")
                    self._record_with_arecord(temporary_path)
                else:
                    try:
                        LOGGER.info(
                            "USB audio recording started: sample_rate=%d mono 16-bit duration=%.1fs",
                            self.sample_rate,
                            self.duration_seconds,
                        )
                        recording = sd.rec(
                            frame_count,
                            samplerate=self.sample_rate,
                            channels=CHANNELS,
                            dtype="int16",
                            blocking=False,
                        )
                        # The stream is open; servo taps may now begin.
                        self._capture_started.set()
                        sd.wait()
                        self._write_wave(temporary_path, recording.tobytes())
                    except Exception:
                        # PortAudio may import correctly but reject the USB
                        # microphone's native sample rate inside Docker. ALSA's
                        # plughw device can resample it transparently.
                        self._capture_started.clear()
                        LOGGER.warning(
                            "sounddevice recording failed; falling back to arecord",
                            exc_info=True,
                        )
                        self._record_with_arecord(temporary_path)

            # Chỉ thay record.wav sau khi tệp tạm đã được ghi hoàn chỉnh.
            temporary_path.replace(self.output_path)

            LOGGER.info("Audio recording saved: %s", self.output_path)
            return self.output_path
        finally:
            self._recording_lock.release()

    def _write_wave(self, path: Path, audio_bytes: bytes) -> None:
        """Ghi dữ liệu PCM 16-bit mono vào một tệp WAV hoàn chỉnh."""
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_bytes)

    def _record_with_arecord(self, output_path: Path) -> None:
        """Ghi đồng bộ bằng ALSA khi sounddevice chưa khả dụng."""
        command = [
            "arecord",
            "-q",
            "-D",
            self.alsa_device,
            "-f",
            "S16_LE",
            "-r",
            str(self.sample_rate),
            "-c",
            str(CHANNELS),
            "-d",
            str(int(self.duration_seconds)),
            str(output_path),
        ]
        try:
            process = subprocess.Popen(
                command,
            )
            self._capture_started.set()
            return_code = process.wait(timeout=self.duration_seconds + 5.0)
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            raise RuntimeError("Could not record from the USB microphone") from exc
