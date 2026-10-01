#!/usr/bin/env python3
"""Unit tests for the browser-facing camera preview address."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from services.camera_preview_server import _preview_display_host  # noqa: E402


def test_explicit_bind_host_is_preserved() -> None:
    assert _preview_display_host("192.168.1.20") == "192.168.1.20"
    assert _preview_display_host("camera.local") == "camera.local"


def test_wildcard_bind_uses_lan_address() -> None:
    probe = Mock()
    probe.getsockname.return_value = ("192.168.1.42", 54321)
    with patch("services.camera_preview_server.socket.socket", return_value=probe):
        assert _preview_display_host("0.0.0.0") == "192.168.1.42"
    probe.connect.assert_called_once_with(("192.0.2.1", 9))
    probe.close.assert_called_once()


def test_wildcard_bind_falls_back_to_loopback_without_network() -> None:
    probe = Mock()
    probe.connect.side_effect = OSError("network unavailable")
    with (
        patch("services.camera_preview_server.socket.socket", return_value=probe),
        patch(
            "services.camera_preview_server.socket.getaddrinfo",
            side_effect=OSError("hostname unavailable"),
        ),
    ):
        assert _preview_display_host("0.0.0.0") == "127.0.0.1"
    probe.close.assert_called_once()


def test_wildcard_bind_handles_blocked_socket_creation() -> None:
    with (
        patch(
            "services.camera_preview_server.socket.socket",
            side_effect=PermissionError("socket blocked"),
        ),
        patch(
            "services.camera_preview_server.socket.getaddrinfo",
            side_effect=OSError("hostname unavailable"),
        ),
    ):
        assert _preview_display_host("0.0.0.0") == "127.0.0.1"


def main() -> None:
    test_explicit_bind_host_is_preserved()
    test_wildcard_bind_uses_lan_address()
    test_wildcard_bind_falls_back_to_loopback_without_network()
    test_wildcard_bind_handles_blocked_socket_creation()
    print("Camera preview address tests: PASS")


if __name__ == "__main__":
    main()
