import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from capture.frame_metadata import save_sample_frame


class FrameMetadataTests(unittest.TestCase):
    def test_saves_frame_and_metadata_sidecar(self) -> None:
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        timestamp = datetime(2026, 10, 4, 10, 30, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path, metadata_path = save_sample_frame(
                frame,
                Path(temporary_directory) / "sample.png",
                frame_number=7,
                capture_fps=58.25,
                monitor_index=2,
                monitor_bounds={"left": 1920, "top": 0, "width": 2560, "height": 1440},
                capture_region={"left": 2000, "top": 100, "width": 640, "height": 480},
                timestamp=timestamp,
            )

            self.assertTrue(image_path.is_file())
            self.assertTrue(metadata_path.is_file())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(metadata["timestamp"], timestamp.isoformat())
        self.assertEqual(metadata["frame_number"], 7)
        self.assertEqual(metadata["capture_fps"], 58.25)
        self.assertEqual(metadata["monitor_index"], 2)
        self.assertEqual(metadata["monitor_bounds"]["width"], 2560)
        self.assertEqual(metadata["capture_region"]["left"], 2000)
        self.assertEqual(
            metadata["frame_resolution"], {"width": 32, "height": 24}
        )

    def test_rejects_timezone_naive_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "timezone information"):
                save_sample_frame(
                    np.zeros((4, 4, 3), dtype=np.uint8),
                    Path(temporary_directory) / "sample.png",
                    frame_number=1,
                    capture_fps=1.0,
                    monitor_index=1,
                    monitor_bounds={"left": 0, "top": 0, "width": 4, "height": 4},
                    capture_region={"left": 0, "top": 0, "width": 4, "height": 4},
                    timestamp=datetime(2026, 10, 4),
                )

    def test_capture_loop_saves_one_frame_and_stops_at_frame_limit(self) -> None:
        from app.config import CaptureConfig
        from capture.screen_capture import run_live_capture

        class FakeMss:
            monitors = [
                {"left": 0, "top": 0, "width": 4, "height": 4},
                {"left": 0, "top": 0, "width": 4, "height": 4},
            ]

            def __init__(self) -> None:
                self.grab_count = 0
                self.last_bounds: dict[str, int] | None = None

            def grab(self, bounds: dict[str, int]) -> np.ndarray:
                self.grab_count += 1
                self.last_bounds = bounds
                return np.zeros((4, 4, 4), dtype=np.uint8)

            def close(self) -> None:
                pass

        fake_capture = FakeMss()
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch("capture.screen_capture.mss", return_value=fake_capture):
                run_live_capture(
                    CaptureConfig(max_fps=1000),
                    output_dir=Path(temporary_directory),
                    save_first_frame=True,
                    max_frames=3,
                    show_preview=False,
                )

            image_files = list(Path(temporary_directory).glob("*.png"))
            metadata_files = list(Path(temporary_directory).glob("*.json"))
            self.assertEqual(fake_capture.grab_count, 3)
            self.assertEqual(
                fake_capture.last_bounds,
                {"left": 0, "top": 0, "width": 4, "height": 4},
            )
            self.assertEqual(len(image_files), 1)
            self.assertEqual(len(metadata_files), 1)
            metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
            self.assertEqual(metadata["frame_number"], 1)
            self.assertEqual(metadata["monitor_index"], 1)
            self.assertEqual(metadata["frame_resolution"], {"width": 4, "height": 4})
            self.assertGreater(metadata["capture_fps"], 0)


if __name__ == "__main__":
    unittest.main()
