"""Debounced GPIO limit switch wrapper."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable


LOGGER = logging.getLogger(__name__)
Callback = Callable[[], None]


class LimitSwitch:
    """Expose a gpiozero Button through guarded callbacks and mock controls."""

    def __init__(
        self,
        gpio_pin: int = 5,
        bounce_time: float = 0.05,
        active_low: bool = True,
        mock: bool = False,
    ) -> None:
        self.gpio_pin = gpio_pin
        self.bounce_time = bounce_time
        self.active_low = active_low
        self.mock = mock
        self.is_connected = False
        self._button = None
        self._pressed = False
        self._lock = threading.RLock()
        self._on_pressed: Callback | None = None
        self._on_released: Callback | None = None

    def initialize(self) -> None:
        """Initialize BCM GPIO with an internal pull-up or pull-down resistor."""
        if self.mock:
            self.is_connected = True
            LOGGER.info(
                "Limit switch initialized in mock mode on BCM GPIO%s active_low=%s",
                self.gpio_pin,
                self.active_low,
            )
            return
        try:
            from gpiozero import Button

            self._button = Button(
                self.gpio_pin,
                pull_up=self.active_low,
                bounce_time=self.bounce_time,
            )
            with self._lock:
                self._pressed = bool(self._button.is_pressed)
            self._button.when_pressed = self._handle_pressed
            self._button.when_released = self._handle_released
            self.is_connected = True
            LOGGER.info(
                "Limit switch initialized on BCM GPIO%s, debounce %.3fs, active_low=%s, initial=%s",
                self.gpio_pin,
                self.bounce_time,
                self.active_low,
                "PRESSED" if self._pressed else "RELEASED",
            )
        except Exception:
            self.is_connected = False
            LOGGER.exception("Limit switch initialization failed")
            raise

    def register_on_pressed(self, callback: Callback) -> None:
        """Register a fast callback invoked after a debounced press."""
        with self._lock:
            self._on_pressed = callback

    def register_on_released(self, callback: Callback) -> None:
        """Register a callback invoked after a debounced release."""
        with self._lock:
            self._on_released = callback

    def _run_callback(self, callback: Callback | None, event: str) -> None:
        LOGGER.info("Limit switch %s", event)
        if callback is None:
            return
        try:
            callback()
        except Exception:
            LOGGER.exception("Limit switch %s callback failed", event)

    def _handle_pressed(self) -> None:
        with self._lock:
            if self._pressed:
                LOGGER.debug("Duplicate pressed state ignored")
                return
            self._pressed = True
            callback = self._on_pressed
        self._run_callback(callback, "pressed")

    def _handle_released(self) -> None:
        with self._lock:
            if not self._pressed:
                LOGGER.debug("Duplicate released state ignored")
                return
            self._pressed = False
            callback = self._on_released
        self._run_callback(callback, "released")

    def is_pressed(self) -> bool:
        """Return the current logical switch state."""
        if not self.mock and self._button is not None:
            return bool(self._button.is_pressed)
        with self._lock:
            return self._pressed

    def simulate_press(self) -> None:
        """Trigger a mock press for tests and Enter-key operation."""
        if not self.mock:
            raise RuntimeError("simulate_press is available only in mock mode")
        self._handle_pressed()

    def simulate_release(self) -> None:
        """Trigger a mock release for tests."""
        if not self.mock:
            raise RuntimeError("simulate_release is available only in mock mode")
        self._handle_released()

    def close(self) -> None:
        """Disable callbacks and release gpiozero resources."""
        with self._lock:
            self._on_pressed = None
            self._on_released = None
            try:
                if self._button is not None:
                    self._button.when_pressed = None
                    self._button.when_released = None
                    self._button.close()
            except Exception:
                LOGGER.exception("Limit switch cleanup failed")
            finally:
                self._button = None
                self.is_connected = False
                LOGGER.info("Limit switch closed")
