import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.config import CaptureConfig, CaptureRegion
from data.independent_capture import (
    IndependentCaptureConfig,
    Purpose,
    initialize_independent_dataset,
    record_independent_session,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class FakeCapture:
    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self.bounds = {
            "left": 0,
            "top": 0,
            "width": config.region.width if config.region else 10,
            "height": config.region.height if config.region else 8,
        }
        self.monitor_bounds = {
            "left": 0,
            "top": 0,
            "width": 100,
            "height": 80,
        }

    def __enter__(self) -> "FakeCapture":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def capture(self) -> np.ndarray:
        return np.zeros((self.bounds["height"], self.bounds["width"], 3), dtype=np.uint8)


class IndependentCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "independent_eval"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _record(self, purpose: Purpose, session_id: str):
        clock = FakeClock()
        config = IndependentCaptureConfig(
            capture=CaptureConfig(
                monitor_index=1,
                max_fps=30,
                region=CaptureRegion(0, 0, 10, 8),
            ),
            purpose=purpose,
            session_id=session_id,
            sample_interval_seconds=0.5,
            max_duration_seconds=1.1,
            notes=f"{purpose} test recording",
            output_root=self.root,
        )
        return record_independent_session(
            config,
            capture_factory=FakeCapture,
            clock=clock,
            sleeper=clock.sleep,
            show_preview=False,
        )

    def test_validation_and_test_use_fixed_separate_splits(self) -> None:
        validation = self._record("validation", "validation_sample")
        test = self._record("test", "test_sample")

        self.assertEqual((validation.split, test.split), ("val", "test"))
        self.assertEqual(validation.frame_count, 3)
        self.assertEqual(test.frame_count, 3)
        self.assertEqual(
            len(list((self.root / "images" / "val").glob("*.png"))), 3
        )
        self.assertEqual(
            len(list((self.root / "images" / "test").glob("*.png"))), 3
        )
        self.assertEqual(
            list((self.root / "images" / "train").iterdir()),
            [],
        )
        manifest_rows = [
            json.loads(line)
            for line in (self.root / "metadata" / "manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual({row["split"] for row in manifest_rows}, {"val", "test"})
        self.assertEqual(
            {row["purpose"] for row in manifest_rows}, {"validation", "test"}
        )

    def test_session_metadata_and_frame_sidecar_are_complete(self) -> None:
        report = self._record("validation", "validation_metadata")
        session = json.loads(report.session_metadata.read_text(encoding="utf-8"))
        self.assertEqual(session["session_id"], report.session_id)
        self.assertEqual(session["purpose"], "validation")
        self.assertEqual(session["source"], "live_screen_capture")
        self.assertIsNone(session["source_video"])
        self.assertEqual(session["split"], "val")
        self.assertEqual(session["source_resolution"], {"width": 10, "height": 8})
        self.assertEqual(session["capture_fps"]["configured_max"], 30)
        self.assertEqual(session["capture_interval_seconds"], 0.5)
        self.assertEqual(session["frame_count"], 3)
        self.assertTrue(session["timestamp"])
        self.assertEqual(session["notes"], "validation test recording")
        self.assertFalse(session["labels_generated"])
        self.assertTrue(session["complete"])

        sidecars = list((self.root / "images" / "val").glob("*.json"))
        sidecar = json.loads(sidecars[0].read_text(encoding="utf-8"))
        self.assertEqual(sidecar["source_session"], report.session_id)
        self.assertEqual(sidecar["purpose"], "validation")
        self.assertEqual(sidecar["split"], "val")
        self.assertEqual(sidecar["source"], "live_screen_capture")
        self.assertIsNone(sidecar["source_video"])
        self.assertEqual(sidecar["frame_resolution"], {"width": 10, "height": 8})
        self.assertTrue(sidecar["timestamp"])

    def test_existing_session_id_cannot_be_reused_or_overwritten(self) -> None:
        self._record("validation", "same_id")
        with self.assertRaises(FileExistsError):
            self._record("test", "same_id")

    def test_dataset_structure_and_config_are_two_class_and_split_isolated(self) -> None:
        initialize_independent_dataset(self.root)
        config = (self.root / "dataset.yaml").read_text(encoding="utf-8")
        self.assertIn("  0: player", config)
        self.assertIn("  1: enemy", config)
        self.assertEqual(len(list((self.root / "images" / "train").iterdir())), 0)
        for split in ("train", "val", "test"):
            self.assertTrue((self.root / "images" / split).is_dir())
            self.assertTrue((self.root / "labels" / split).is_dir())

    def test_capture_root_overlapping_legacy_or_future_dataset_is_rejected(self) -> None:
        for protected in (
            Path("data/dataset"),
            Path("data/dataset_player_enemy_v2"),
            Path("data"),
        ):
            with self.subTest(protected=protected):
                with self.assertRaisesRegex(ValueError, "overlaps protected dataset"):
                    initialize_independent_dataset(protected)

    def test_capture_refuses_legacy_root_before_creating_session_files(self) -> None:
        config = IndependentCaptureConfig(
            capture=CaptureConfig(),
            purpose="validation",
            session_id="blocked_legacy_write",
            output_root=Path("data/dataset"),
        )
        with self.assertRaisesRegex(ValueError, "overlaps protected dataset"):
            record_independent_session(config, show_preview=False)
        self.assertFalse(
            (Path("data/dataset/metadata/validation/blocked_legacy_write")).exists()
        )

    def test_invalid_purpose_or_nonunique_session_path_is_rejected(self) -> None:
        invalid = IndependentCaptureConfig(
            capture=CaptureConfig(),
            purpose="train",
            session_id="bad-purpose",
            output_root=self.root,
        )
        with self.assertRaisesRegex(ValueError, "validation or test"):
            invalid.validate()
        with self.assertRaisesRegex(ValueError, "Session ID"):
            IndependentCaptureConfig(
                capture=CaptureConfig(),
                purpose="test",
                session_id="../unsafe",
                output_root=self.root,
            ).validate()


if __name__ == "__main__":
    unittest.main()
