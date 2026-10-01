"""Single-worker camera-to-classification workflow."""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hardware.audio_recorder import AudioRecorder
from hardware.camera_service import CameraService
from hardware.lcd_display import LCDDisplay, LCD_IDLE_LINE_1, LCD_IDLE_LINE_2
from hardware.servo_controller import ServoController
from inference.audio_classifier import TorchAudioClassifier
from inference.classifier import CLASS_NAMES, KerasImageClassifier
from inference.exceptions import CameraBlurError, CameraError, ModelError
from inference.onnx_image_classifier import OnnxImageClassifier
from inference.presence_detector import EmptySceneDetector
from inference.result import ClassificationResult
from inference.torch_image_classifier import TorchImageClassifier
from .result_store import LatestResultStore


LOGGER = logging.getLogger(__name__)


class NoDurianDetectedError(Exception):
    """Raised when the camera scene still matches the empty reference."""

class CaptureClassifyService:
    """Serialize switch-triggered classification jobs through one worker."""

    def __init__(
        self,
        camera: CameraService,
        classifier: KerasImageClassifier | TorchImageClassifier | OnnxImageClassifier,
        lcd: LCDDisplay,
        result_display_seconds: float,
        error_display_seconds: float,
        delete_image_after_inference: bool,
        servo_controller: ServoController | None = None,
        audio_recorder: AudioRecorder | None = None,
        audio_classifier: TorchAudioClassifier | None = None,
        images_per_job: int = 3,
        image_interval_seconds: float = 3.0,
        fusion_image_weight: float = 0.6,
        fusion_audio_weight: float = 0.4,
        presence_detector: EmptySceneDetector | None = None,
        result_store: LatestResultStore | None = None,
    ) -> None:
        self.camera = camera
        self.classifier = classifier
        self.lcd = lcd
        self.result_display_seconds = result_display_seconds
        self.error_display_seconds = error_display_seconds
        self.delete_image_after_inference = delete_image_after_inference
        # Hai module mới là tùy chọn để các luồng kiểm thử cũ vẫn hoạt động nguyên vẹn.
        self.servo_controller = servo_controller
        self.audio_recorder = audio_recorder
        self.audio_classifier = audio_classifier
        self.images_per_job = images_per_job
        self.image_interval_seconds = image_interval_seconds
        # Hai trọng số áp dụng cho toàn bộ vector xác suất của ba lớp.
        self.fusion_image_weight = fusion_image_weight
        self.fusion_audio_weight = fusion_audio_weight
        self.presence_detector = presence_detector
        self.result_store = result_store
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._shutdown_event = threading.Event()
        self._busy_lock = threading.Lock()
        self._busy = False
        self._worker: threading.Thread | None = None

    def initialize(self) -> None:
        """Start exactly one background worker."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._worker_loop, name="capture-worker", daemon=True)
        self._worker.start()
        LOGGER.info("Capture/classify worker started")

    def on_switch_pressed(self) -> None:
        """Accept at most one job and return immediately to the GPIO callback."""
        with self._busy_lock:
            if self._busy:
                LOGGER.warning("Switch press ignored because pipeline is busy")
                return
            self._busy = True
        try:
            self._queue.put_nowait("capture")
            if self.result_store is not None:
                self.result_store.begin_job()
            LOGGER.info("Capture job queued")
        except queue.Full:
            with self._busy_lock:
                self._busy = False
            LOGGER.warning("Switch press ignored because queue is full")

    def is_busy(self) -> bool:
        """Return whether a job is queued or running."""
        with self._busy_lock:
            return self._busy

    def _worker_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                job = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                if job is None:
                    return
                self.process_once()
            except Exception:
                LOGGER.exception("Unhandled worker error; worker will continue")
            finally:
                self._queue.task_done()

    def process_once(self) -> None:
        """Chạy đồng thời nhánh ba ảnh và nhánh audio/servo rồi fusion."""
        total_started = time.perf_counter()
        image_paths: list[str] = []
        inference_succeeded = False
        try:
            image_results: list[ClassificationResult] = []

            # Model audio/servo và model ảnh dùng hai worker độc lập. Nhánh audio
            # bắt đầu trước để microphone thu trọn chuỗi gõ trong khi camera chụp.
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio-pipeline") as executor:
                audio_future = executor.submit(self._run_audio_pipeline)
                capture_started = time.monotonic()
                for image_number in range(1, self.images_per_job + 1):
                    if image_number > 1:
                        target_time = capture_started + (
                            (image_number - 1) * self.image_interval_seconds
                        )
                        delay = max(0.0, target_time - time.monotonic())
                        if self._shutdown_event.wait(delay):
                            raise RuntimeError("Shutdown requested during image capture")
                    self.lcd.display_message(
                        "CAPTURING",
                        f"IMAGE {image_number}/{self.images_per_job}",
                    )
                    if self.result_store is not None:
                        self.result_store.update_status(
                            "capturing",
                            f"Đang chụp ảnh {image_number}/{self.images_per_job}",
                        )
                    image_path = self.camera.capture_image()
                    image_paths.append(image_path)
                    present: bool | None = None
                    presence_score: float | None = None
                    if self.presence_detector is not None:
                        present, presence_score = self.presence_detector.detect(image_path)
                    web_image_index = (
                        self.result_store.add_image(
                            image_path,
                            present=present,
                            presence_score=presence_score,
                        )
                        if self.result_store is not None
                        else None
                    )
                    if present is False:
                        continue
                    self.lcd.display_message(
                        "ANALYZING IMAGE",
                        f"IMAGE {image_number}/{self.images_per_job}",
                    )
                    if self.result_store is not None:
                        self.result_store.update_status(
                            "analyzing",
                            f"Đang phân tích ảnh {image_number}/{self.images_per_job}",
                        )
                    image_result = self.classifier.predict(image_path)
                    image_results.append(image_result)
                    if self.result_store is not None and web_image_index is not None:
                        self.result_store.set_image_result(web_image_index, image_result)
                # Đây là điểm đồng bộ bắt buộc: tap_sequence() trong nhánh audio
                # phải hoàn tất đủ ba nhịp trước khi kết quả được đưa lên LCD.
                audio_result = audio_future.result()

            # Model độ chín là bộ phân loại đóng và luôn chọn một trong ba lớp.
            # Không cho model kết luận nếu cả ba ảnh vẫn giống khay trống.
            if not image_results:
                raise NoDurianDetectedError

            # Tính kết quả một lần và trả ngay, không kiểm tra ngưỡng để chạy lại.
            result = self._fuse_results(audio_result, image_results)
            inference_succeeded = True
            self.lcd.display_result(result.class_name, result.confidence)
            total_ms = (time.perf_counter() - total_started) * 1000.0
            LOGGER.info(
                "Fusion completed images=%s audio=%s index=%s class=%s confidence=%.6f "
                "probabilities=%s inference_ms=%.2f total_ms=%.2f",
                image_paths,
                audio_result.probabilities if audio_result is not None else None,
                result.class_index,
                result.class_name,
                result.confidence,
                result.probabilities,
                result.inference_time_ms,
                total_ms,
            )
            if self.result_store is not None:
                self.result_store.complete(result, total_ms)
            # Hiển thị kết quả đủ thời gian cấu hình rồi trở về màn hình chờ.
            self._shutdown_event.wait(self.result_display_seconds)
        except NoDurianDetectedError:
            LOGGER.warning("No durian detected in the camera ROI")
            if self.result_store is not None:
                self.result_store.fail(
                    "no_durian",
                    "Không phát hiện sầu riêng trong vùng kiểm tra",
                    (time.perf_counter() - total_started) * 1000.0,
                )
            self.lcd.display_message("NO DURIAN", "PLACE DURIAN")
            self._shutdown_event.wait(self.error_display_seconds)
        except CameraBlurError:
            LOGGER.warning("All camera frames were too blurry", exc_info=True)
            if self.result_store is not None:
                self.result_store.fail(
                    "error",
                    "Ảnh quá mờ, hãy đặt lại sầu riêng",
                    (time.perf_counter() - total_started) * 1000.0,
                )
            self.lcd.display_message("IMAGE TOO BLURRY", "MOVE OBJECT BACK")
            self._shutdown_event.wait(self.error_display_seconds)
        except CameraError:
            LOGGER.exception("Camera pipeline failure")
            if self.result_store is not None:
                self.result_store.fail(
                    "error",
                    "Không thể chụp ảnh từ camera",
                    (time.perf_counter() - total_started) * 1000.0,
                )
            self.lcd.display_message("Error", "Camera failed")
            self._shutdown_event.wait(self.error_display_seconds)
        except ModelError:
            LOGGER.exception("Model pipeline failure")
            if self.result_store is not None:
                self.result_store.fail(
                    "error",
                    "Model AI xử lý thất bại",
                    (time.perf_counter() - total_started) * 1000.0,
                )
            self.lcd.display_message("Error", "Model failed")
            self._shutdown_event.wait(self.error_display_seconds)
        except Exception:
            LOGGER.exception("Unexpected pipeline failure")
            if self.result_store is not None:
                self.result_store.fail(
                    "error",
                    "Hệ thống xử lý thất bại",
                    (time.perf_counter() - total_started) * 1000.0,
                )
            self.lcd.display_message("Error", "System failed")
            self._shutdown_event.wait(self.error_display_seconds)
        finally:
            if inference_succeeded and self.delete_image_after_inference:
                for image_path in image_paths:
                    try:
                        Path(image_path).unlink(missing_ok=True)
                        LOGGER.info(
                            "Deleted captured image after successful inference: %s",
                            image_path,
                        )
                    except OSError:
                        LOGGER.exception("Failed to delete captured image: %s", image_path)
            if not self._shutdown_event.is_set():
                self.lcd.display_message(LCD_IDLE_LINE_1, LCD_IDLE_LINE_2)
            with self._busy_lock:
                self._busy = False

    def _run_audio_pipeline(self) -> ClassificationResult | None:
        """Record three Servo 1 strikes and run audio inference beside image inference."""
        audio_path = self._record_audio_during_servo_strike()
        if audio_path is None:
            return None
        if self.result_store is not None:
            self.result_store.set_audio(audio_path)
        if self.audio_classifier is None:
            return None
        try:
            audio_result = self.audio_classifier.predict(audio_path)
            if self.result_store is not None:
                self.result_store.set_audio(audio_path, audio_result)
            LOGGER.info(
                "Audio model: class=%s confidence=%.2f%% probabilities_pct=%s",
                audio_result.class_name,
                audio_result.confidence * 100.0,
                self._probabilities_as_percentages(audio_result.probabilities),
            )
            return audio_result
        except ModelError:
            LOGGER.exception("Audio inference failed; continuing camera/AI pipeline")
            return None

    def _record_audio_during_servo_strike(self) -> Path | None:
        """Mở micro, xác nhận đang thu rồi mới gõ Servo 1 ba lần."""
        if self.audio_recorder is None:
            if self.servo_controller is not None:
                self.lcd.display_message("CHECKING", "SERVOS TAPPING")
                self.servo_controller.tap_sequence()
            return None

        self.lcd.display_message("RECORDING AUDIO", "SERVOS TAPPING")
        if self.result_store is not None:
            self.result_store.update_status(
                "recording",
                "Đang ghi âm và gõ kiểm tra",
            )
        recording_result: list[Path] = []
        recording_error: list[Exception] = []

        def record() -> None:
            try:
                recording_result.append(self.audio_recorder.record_audio())
            except Exception as exc:
                recording_error.append(exc)

        recording_thread = threading.Thread(
            target=record,
            name="audio-recording",
            daemon=True,
        )
        recording_thread.start()

        recording_started = self.audio_recorder.wait_until_recording_started(
            timeout=2.0
        )
        if recording_started:
            LOGGER.info("Microphone recording started; running three Servo 1 strikes")
            if self.servo_controller is not None:
                self.servo_controller.tap_sequence()
        else:
            # A microphone failure must never suppress the physical tap cycle.
            # Audio inference can be skipped, while image inference and Servo 1
            # continue through the normal device workflow.
            LOGGER.error(
                "Microphone did not start within 2 seconds; running three strikes without audio"
            )
            if self.servo_controller is not None:
                self.servo_controller.tap_sequence()

        recording_thread.join(timeout=self.audio_recorder.duration_seconds + 6.0)
        if recording_thread.is_alive():
            LOGGER.error("Audio recording thread did not finish in time")
            return None
        if recording_error:
            LOGGER.error(
                "Audio recording failed; continuing image pipeline",
                exc_info=(
                    type(recording_error[0]),
                    recording_error[0],
                    recording_error[0].__traceback__,
                ),
            )
            self.lcd.display_message("RECORDING ERROR", "USING IMAGE AI")
            return None
        return recording_result[0] if recording_result else None

    def _fuse_results(
        self,
        audio_result: ClassificationResult | None,
        image_results: list[ClassificationResult],
    ) -> ClassificationResult:
        """Trung bình các ảnh rồi fusion đủ ba lớp với xác suất audio."""
        if not image_results:
            raise ModelError("No image results are available for fusion")

        class_count = len(CLASS_NAMES)
        all_results = [*image_results]
        if audio_result is not None:
            all_results.append(audio_result)
        if any(len(result.probabilities) != class_count for result in all_results):
            raise ModelError("Model probability count does not match configured classes")

        # Nhiều lần chụp vẫn chỉ đại diện cho một model ảnh, tránh làm tăng trọng
        # số model ảnh so với model âm thanh.
        image_probabilities = [
            sum(result.probabilities[class_index] for result in image_results)
            / len(image_results)
            for class_index in range(class_count)
        ]
        image_class_index = max(
            range(class_count), key=image_probabilities.__getitem__
        )
        # Lấy confidence cao nhất của từng ảnh rồi cộng và chia trung bình.
        image_peak_confidences = [
            max(float(probability) for probability in result.probabilities)
            for result in image_results
        ]
        image_confidence = sum(image_peak_confidences) / len(image_peak_confidences)
        LOGGER.info(
            "Image model average: class=%s peak_confidence_avg=%.2f%% "
            "peak_confidences_pct=%s probabilities_pct=%s images=%s",
            CLASS_NAMES[image_class_index],
            image_confidence * 100.0,
            [round(value * 100.0, 2) for value in image_peak_confidences],
            self._probabilities_as_percentages(image_probabilities),
            len(image_results),
        )

        if audio_result is None:
            # Không tự bịa xác suất audio khi micro/model âm thanh lỗi.
            fused_probabilities = [float(value) for value in image_probabilities]
            LOGGER.warning("Audio result unavailable; using image probabilities only")
        else:
            fused_probabilities = [
                self.fusion_image_weight * image_probability
                + self.fusion_audio_weight * float(audio_probability)
                for image_probability, audio_probability in zip(
                    image_probabilities, audio_result.probabilities
                )
            ]

        model_class_index = max(
            range(class_count), key=fused_probabilities.__getitem__
        )
        model_class_name = CLASS_NAMES[model_class_index]
        model_confidence = float(fused_probabilities[model_class_index])
        inference_time_ms = sum(
            result.inference_time_ms for result in image_results
        ) + (audio_result.inference_time_ms if audio_result is not None else 0.0)
        if audio_result is not None:
            LOGGER.info(
                "Fusion 3-class %.0f/%.0f: image_peak_avg=%.2f%% "
                "image_pct=%s audio_pct=%s final_pct=%s result=%s confidence=%.2f%%",
                self.fusion_image_weight * 100.0,
                self.fusion_audio_weight * 100.0,
                image_confidence * 100.0,
                self._probabilities_as_percentages(image_probabilities),
                self._probabilities_as_percentages(audio_result.probabilities),
                self._probabilities_as_percentages(fused_probabilities),
                model_class_name,
                model_confidence * 100.0,
            )
        return ClassificationResult(
            class_index=model_class_index,
            class_name=model_class_name,
            confidence=model_confidence,
            inference_time_ms=inference_time_ms,
            probabilities=[float(value) for value in fused_probabilities],
        )

    @staticmethod
    def _probabilities_as_percentages(probabilities: list[float]) -> dict[str, float]:
        """Ghép tên lớp với phần trăm làm tròn để log dễ đọc."""
        return {
            class_name: round(float(probability) * 100.0, 2)
            for class_name, probability in zip(CLASS_NAMES, probabilities)
        }

    def shutdown(self) -> None:
        """Stop accepting work and join the worker cleanly."""
        self._shutdown_event.set()
        if self._worker is not None and self._worker.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._worker.join(timeout=10.0)
            if self._worker.is_alive():
                LOGGER.warning("Worker did not stop within timeout")
        self._worker = None
        LOGGER.info("Capture/classify worker stopped")
