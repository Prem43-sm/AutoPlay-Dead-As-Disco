import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np

from app.config import CaptureConfig
from data.dataset_tools import (
    CLASS_NAMES,
    DatasetCaptureConfig,
    assign_session_split,
    create_dataset_structure,
    frame_difference,
    frame_filename,
    is_duplicate_frame,
    parse_yolo_labels,
    record_dataset_session,
    render_dataset_preview,
    validate_dataset,
    validate_yolo_label,
    YoloLabel,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class FakeCapture:
    def __init__(self, _config: CaptureConfig, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._index = 0
        self.monitor_bounds = {"left": 0, "top": 0, "width": 16, "height": 12}
        self.bounds = self.monitor_bounds.copy()

    def __enter__(self) -> "FakeCapture":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        pass

    def capture(self) -> np.ndarray:
        frame = self._frames[min(self._index, len(self._frames) - 1)]
        self._index += 1
        return frame.copy()


class DatasetStructureTests(unittest.TestCase):
    def test_capture_config_rejects_nonfinite_sampling_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            DatasetCaptureConfig(
                capture=CaptureConfig(),
                sample_interval=float("nan"),
                max_samples=1,
                output_dir=Path("dataset"),
            ).validate()

    def test_creates_dataset_paths_without_overwriting_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory) / "dataset"
            create_dataset_structure(dataset_dir)
            yaml_path = dataset_dir / "dataset.yaml"
            original = yaml_path.read_text(encoding="utf-8")
            create_dataset_structure(dataset_dir)

            for split in ("train", "val", "test"):
                self.assertTrue((dataset_dir / "images" / split).is_dir())
                self.assertTrue((dataset_dir / "labels" / split).is_dir())
            self.assertTrue((dataset_dir / "metadata").is_dir())
            self.assertEqual(yaml_path.read_text(encoding="utf-8"), original)
            self.assertEqual(len(CLASS_NAMES), 6)

    def test_frame_filename_has_sequence_and_utc_timestamp(self) -> None:
        timestamp = datetime(2026, 10, 4, 10, 30, tzinfo=timezone.utc)
        self.assertEqual(
            frame_filename(4, timestamp),
            "frame_000004_20261004T103000_000000Z.png",
        )

    def test_split_assignment_is_stable_per_session(self) -> None:
        self.assertEqual(assign_session_split("same-session"), assign_session_split("same-session"))
        self.assertIn(assign_session_split("same-session"), {"train", "val", "test"})


class DeduplicationTests(unittest.TestCase):
    def test_similarity_filters_identical_frames(self) -> None:
        first = np.zeros((40, 80, 3), dtype=np.uint8)
        same = first.copy()
        changed = first.copy()
        changed[10:30, 20:60] = 255

        self.assertEqual(frame_difference(first, same), 0)
        self.assertTrue(is_duplicate_frame(first, same, 2.0))
        self.assertFalse(is_duplicate_frame(first, changed, 2.0))
        self.assertFalse(is_duplicate_frame(None, same, 2.0))


class DatasetRecorderTests(unittest.TestCase):
    def _run_recorder(
        self,
        dataset_dir: Path,
        frames: list[np.ndarray],
        *,
        deduplicate: bool = False,
    ):
        clock = FakeClock()

        def factory(config: CaptureConfig) -> FakeCapture:
            return FakeCapture(config, frames)

        return record_dataset_session(
            DatasetCaptureConfig(
                capture=CaptureConfig(monitor_index=1, max_fps=2),
                sample_interval=1.0,
                max_samples=3,
                output_dir=dataset_dir,
                session_id="test-session",
                split="train",
                deduplicate=deduplicate,
                similarity_threshold=2.0,
            ),
            capture_factory=factory,
            clock=clock,
            sleeper=clock.sleep,
        )

    def test_recorder_obeys_interval_limit_and_writes_manifest_metadata(self) -> None:
        frames = [
            np.full((12, 16, 3), value, dtype=np.uint8)
            for value in (0, 0, 10, 10, 20)
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory) / "dataset"
            report = self._run_recorder(dataset_dir, frames)

            self.assertEqual(report.sampled_frames, 3)
            self.assertEqual(report.captured_frames, 5)
            self.assertEqual(report.saved_frames, 3)
            self.assertEqual(report.split, "train")
            image_paths = list((dataset_dir / "images" / "train").glob("*.png"))
            self.assertEqual(len(image_paths), 3)
            self.assertTrue(report.manifest_path.is_file())
            rows = [
                json.loads(line)
                for line in report.manifest_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(not row["labeled"] and row["labels"] == [] for row in rows))
            self.assertTrue(all(row["source_session"] == "test-session" for row in rows))
            self.assertTrue(all((dataset_dir / row["image_path"]).is_file() for row in rows))
            self.assertTrue(report.session_metadata_path.is_file())
            session_record = dataset_dir.parent / "sessions" / "test-session.json"
            self.assertTrue(session_record.is_file())
            self.assertEqual(len(list((dataset_dir / "metadata" / "test-session").glob("*.png"))), 0)

    def test_recorder_deduplicates_without_labeling_or_overwriting(self) -> None:
        same = np.zeros((12, 16, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory) / "dataset"
            report = self._run_recorder(dataset_dir, [same], deduplicate=True)
            self.assertEqual(report.sampled_frames, 3)
            self.assertEqual(report.saved_frames, 1)
            self.assertEqual(report.duplicate_frames_skipped, 2)


class YoloLabelTests(unittest.TestCase):
    def test_parses_yolo_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.txt"
            path.write_text("0 0.5 0.5 0.2 0.4\n5 0.1 0.2 0.1 0.1\n", encoding="utf-8")
            labels = parse_yolo_labels(path)
        self.assertEqual([label.class_id for label in labels], [0, 5])

    def test_rejects_malformed_label_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.txt"
            path.write_text("0 0.5 0.5 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 5 fields"):
                parse_yolo_labels(path)

    def test_rejects_noninteger_class_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.txt"
            path.write_text("0.5 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "class ID must be an integer"):
                parse_yolo_labels(path)

    def test_validates_class_bounds_and_nonfinite_coordinates(self) -> None:
        classes = {index: name for index, name in enumerate(CLASS_NAMES)}
        self.assertIsNone(validate_yolo_label(YoloLabel(0, 0.5, 0.5, 0.2, 0.4), 100, 100, classes))
        self.assertIn("unknown class", validate_yolo_label(YoloLabel(8, 0.5, 0.5, 0.2, 0.4), 100, 100, classes) or "")
        self.assertIn("outside image", validate_yolo_label(YoloLabel(0, 0.05, 0.5, 0.2, 0.4), 100, 100, classes) or "")
        self.assertIn("finite", validate_yolo_label(YoloLabel(0, float("nan"), 0.5, 0.2, 0.4), 100, 100, classes) or "")


class DatasetValidationAndPreviewTests(unittest.TestCase):
    def test_validator_reports_unlabeled_images_and_split_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory) / "dataset"
            create_dataset_structure(dataset_dir)
            image = np.zeros((20, 30, 3), dtype=np.uint8)
            cv2.imwrite(str(dataset_dir / "images" / "train" / "unlabeled.png"), image)

            report = validate_dataset(dataset_dir)

        self.assertTrue(report.valid)
        self.assertEqual(report.split_image_counts["train"], 1)
        self.assertEqual(report.total_image_count, 1)
        self.assertEqual(report.unlabeled_image_count, 1)
        self.assertEqual(report.no_target_class_image_count, 0)
        self.assertIn("Missing label", report.warnings[0])

    def test_validator_reports_dataset_quality_statistics_and_session_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory) / "dataset"
            create_dataset_structure(dataset_dir)
            image = np.zeros((20, 30, 3), dtype=np.uint8)
            train_dir = dataset_dir / "images" / "train"
            val_dir = dataset_dir / "images" / "val"
            train_first = train_dir / "gameplay_frame_000001_20261004T000000_000000Z.png"
            train_second = train_dir / "gameplay_frame_000002_20261004T000001_000000Z.png"
            val_third = val_dir / "gameplay_frame_000003_20261004T000002_000000Z.png"
            for image_path in (train_first, train_second, val_third):
                cv2.imwrite(str(image_path), image)
            (dataset_dir / "labels" / "train" / f"{train_first.stem}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n",
                encoding="utf-8",
            )
            (dataset_dir / "labels" / "train" / f"{train_second.stem}.txt").write_text(
                "",
                encoding="utf-8",
            )
            (dataset_dir / "labels" / "val" / f"{val_third.stem}.txt").write_text(
                "",
                encoding="utf-8",
            )

            report = validate_dataset(dataset_dir)

        self.assertFalse(report.valid)
        self.assertEqual(report.total_image_count, 3)
        self.assertEqual(report.labeled_image_count, 3)
        self.assertEqual(report.multi_class_image_count, 1)
        self.assertEqual(report.no_target_class_image_count, 2)
        self.assertEqual(report.exact_duplicate_image_count, 2)
        self.assertEqual(report.near_duplicate_pair_count, 2)
        self.assertEqual(report.invalid_label_count, 0)
        self.assertEqual(report.session_split_leaks, ("gameplay",))
        self.assertEqual(report.label_object_counts["player"], 1)
        self.assertTrue(any("split across" in error for error in report.errors))

    def test_validator_detects_bad_labels_corruption_and_duplicate_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory) / "dataset"
            create_dataset_structure(dataset_dir)
            train_image = dataset_dir / "images" / "train" / "same.png"
            val_image = dataset_dir / "images" / "val" / "same.png"
            cv2.imwrite(str(train_image), np.zeros((20, 30, 3), dtype=np.uint8))
            cv2.imwrite(str(val_image), np.zeros((20, 30, 3), dtype=np.uint8))
            (dataset_dir / "labels" / "train" / "same.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n99 0.5 0.5 1.2 0.5\n",
                encoding="utf-8",
            )
            (dataset_dir / "images" / "test" / "broken.png").write_bytes(b"broken")
            (dataset_dir / "labels" / "val" / "orphan.txt").write_text("", encoding="utf-8")

            report = validate_dataset(dataset_dir)

        self.assertFalse(report.valid)
        joined = "\n".join(report.errors)
        self.assertIn("Duplicate image filename", joined)
        self.assertIn("Corrupted or unreadable", joined)
        self.assertIn("unknown class ID", joined)
        self.assertIn("Label has no matching image", joined)
        self.assertEqual(report.class_frame_counts["player"], 1)

    def test_validator_detects_missing_manifest_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory) / "dataset"
            create_dataset_structure(dataset_dir)
            manifest = dataset_dir / "metadata" / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "image_path": "images/train/missing.png",
                        "source_session": "missing-session",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = validate_dataset(dataset_dir)

        self.assertFalse(report.valid)
        self.assertIn("missing image", report.errors[0])

    def test_preview_uses_manual_labels_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory) / "dataset"
            create_dataset_structure(dataset_dir)
            image_path = dataset_dir / "images" / "train" / "sample.png"
            label_path = dataset_dir / "labels" / "train" / "sample.txt"
            cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))
            label_path.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

            preview = render_dataset_preview(image_path, dataset_dir=dataset_dir)

        self.assertEqual(preview.shape, (100, 100, 3))
        self.assertTrue(np.any(np.all(preview == (0, 255, 0), axis=2)))


if __name__ == "__main__":
    unittest.main()
