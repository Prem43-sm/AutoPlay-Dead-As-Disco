import unittest
from unittest.mock import patch

import numpy as np

from app.config import CaptureConfig, CaptureRegion
from capture.screen_capture import ScreenCapture


class FakeMss:
    def __init__(self) -> None:
        self.monitors = [
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
            {"left": 1920, "top": 0, "width": 2560, "height": 1440},
        ]
        self.closed = False
        self.last_bounds: dict[str, int] | None = None

    def grab(self, bounds: dict[str, int]) -> np.ndarray:
        self.last_bounds = bounds
        return np.array([[[10, 20, 30, 255]]], dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


class ScreenCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_capture = FakeMss()
        self.capture_patch = patch(
            "capture.screen_capture.mss", return_value=self.fake_capture
        )
        self.capture_patch.start()
        self.addCleanup(self.capture_patch.stop)

    def test_captures_a_monitor_relative_region(self) -> None:
        capture = ScreenCapture(
            CaptureConfig(
                monitor_index=1,
                region=CaptureRegion(left=10, top=20, width=100, height=200),
            )
        )
        self.addCleanup(capture.close)

        frame = capture.capture()

        self.assertEqual(
            capture.bounds,
            {"left": 1930, "top": 20, "width": 100, "height": 200},
        )
        self.assertEqual(self.fake_capture.last_bounds, capture.bounds)
        self.assertEqual(frame.shape, (1, 1, 3))
        self.assertEqual(frame[0, 0].tolist(), [10, 20, 30])

    def test_rejects_a_region_outside_monitor_and_closes_capture(self) -> None:
        with self.assertRaisesRegex(ValueError, "fit within the selected monitor"):
            ScreenCapture(
                CaptureConfig(
                    monitor_index=1,
                    region=CaptureRegion(left=2500, top=0, width=100, height=100),
                )
            )

        self.assertTrue(self.fake_capture.closed)

    def test_rejects_an_unavailable_monitor_and_closes_capture(self) -> None:
        with self.assertRaisesRegex(ValueError, "Monitor 2 is unavailable"):
            ScreenCapture(CaptureConfig(monitor_index=2))

        self.assertTrue(self.fake_capture.closed)


if __name__ == "__main__":
    unittest.main()
