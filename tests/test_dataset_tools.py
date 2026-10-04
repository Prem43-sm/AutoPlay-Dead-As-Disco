import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np

from app.config import CaptureConfig
from data.dataset_labeler import (
    build_labeling_report,
    get_labeling_progress,
    load_labeling_state,
    pixel_box_to_yolo,
    read_yolo_annotations,
    save_labeling_state,
    write_yolo_annotations,
    yolo_to_pixel_box,
)
from data.dataset_tools import (
    CLASS_NAMES,
    DatasetCaptureConfig,
    DatasetSplit,
    VideoDatasetCaptureConfig,
    assign_session_split,
    create_dataset_structure,
    frame_difference,
    frame_filename,
    extract_video_dataset_session,
    is_duplicate_frame,
    parse_yolo_labels,
    record_dataset_session,
    read_video_metadata,
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


class FakeVideoCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        fps: float = 2.0,
        opened: bool = True,
    ) -> None:
        self._frames = frames
        self._fps = fps
        self._opened = opened
        self._index = 0
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._frames[0].shape[1]) if self._frames else 0.0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._frames[0].shape[0]) if self._frames else 0.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._frames))
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index].copy()
        self._index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


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


class VideoDatasetCaptureTests(unittest.TestCase):
    def _config(
        self,
        video_path: Path,
        dataset_dir: Path,
        *,
        sample_interval: float = 1.0,
        max_frames: int = 20,
        session_id: str = "video-session",
        split: DatasetSplit | None = "val",
        deduplicate: bool = False,
    ) -> VideoDatasetCaptureConfig:
        return VideoDatasetCaptureConfig(
            video_path=video_path,
            sample_interval=sample_interval,
            max_frames=max_frames,
            output_dir=dataset_dir,
            session_id=session_id,
            split=split,
            deduplicate=deduplicate,
        )

    def test_reads_video_metadata_and_releases_capture(self) -> None:
        frames = [np.zeros((12, 16, 3), dtype=np.uint8) for _ in range(5)]
        created: list[FakeVideoCapture] = []

        def factory(_path: str) -> FakeVideoCapture:
            capture = FakeVideoCapture(frames, fps=2.0)
            created.append(capture)
            return capture

        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "sample.mp4"
            video_path.write_bytes(b"fixture")
            metadata = read_video_metadata(video_path, capture_factory=factory)

        self.assertEqual(metadata.duration_seconds, 2.5)
        self.assertEqual(metadata.source_fps, 2.0)
        self.assertEqual((metadata.width, metadata.height), (16, 12))
        self.assertEqual(metadata.total_frames, 5)
        self.assertEqual(metadata.source_video, str(video_path.resolve()))
        self.assertTrue(created[0].released)

    def test_extracts_sampled_frames_with_source_metadata_and_split(self) -> None:
        frames = [
            np.full((12, 16, 3), value, dtype=np.uint8)
            for value in (0, 0, 20, 20, 40)
        ]

        def factory(_path: str) -> FakeVideoCapture:
            return FakeVideoCapture(frames, fps=2.0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "sample.mp4"
            video_path.write_bytes(b"fixture")
            dataset_dir = root / "dataset"
            report = extract_video_dataset_session(
                self._config(video_path, dataset_dir, sample_interval=1.0),
                capture_factory=factory,
            )
            images = sorted((dataset_dir / "images" / "val").glob("*.png"))
            rows = [
                json.loads(line)
                for line in report.manifest_path.read_text(encoding="utf-8").splitlines()
            ]
            session_metadata = json.loads(
                report.session_metadata_path.read_text(encoding="utf-8")
            )

        self.assertEqual(report.session_id, "video-session")
        self.assertEqual(report.split, "val")
        self.assertEqual(report.source_frames_read, 5)
        self.assertEqual(report.sampled_frames, 3)
        self.assertEqual(report.saved_frames, 3)
        self.assertEqual(report.duplicate_frames_skipped, 0)
        self.assertEqual(len(images), 3)
        self.assertEqual(
            [row["source_frame_number"] for row in rows],
            [1, 3, 5],
        )
        self.assertEqual(
            [row["source_timestamp_seconds"] for row in rows],
            [0.0, 1.0, 2.0],
        )
        self.assertTrue(all(row["source_fps"] == 2.0 for row in rows))
        self.assertTrue(all(row["source_resolution"] == {"width": 16, "height": 12} for row in rows))
        self.assertTrue(all(row["dataset_session_id"] == "video-session" for row in rows))
        self.assertTrue(all(row["split"] == "val" and not row["labeled"] for row in rows))
        self.assertFalse(session_metadata["labels_generated"])

    def test_video_extraction_is_bounded_and_deduplicates(self) -> None:
        frames = [np.zeros((12, 16, 3), dtype=np.uint8) for _ in range(20)]

        def factory(_path: str) -> FakeVideoCapture:
            return FakeVideoCapture(frames, fps=2.0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "sample.mp4"
            video_path.write_bytes(b"fixture")
            dataset_dir = root / "dataset"
            report = extract_video_dataset_session(
                self._config(
                    video_path,
                    dataset_dir,
                    sample_interval=0.5,
                    max_frames=3,
                    deduplicate=True,
                ),
                capture_factory=factory,
            )

        self.assertEqual(report.source_frames_read, 3)
        self.assertEqual(report.sampled_frames, 3)
        self.assertEqual(report.saved_frames, 1)
        self.assertEqual(report.duplicate_frames_skipped, 2)

    def test_video_extraction_rejects_missing_or_unopenable_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_dir = root / "dataset"
            missing_path = root / "missing.mp4"
            with self.assertRaises(FileNotFoundError):
                extract_video_dataset_session(
                    self._config(missing_path, dataset_dir),
                    capture_factory=lambda _path: self.fail("must not open missing video"),
                )

            invalid_path = root / "invalid.mp4"
            invalid_path.write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "Could not open"):
                extract_video_dataset_session(
                    self._config(invalid_path, dataset_dir),
                    capture_factory=lambda _path: FakeVideoCapture([], opened=False),
                )

    def test_video_metadata_rejects_invalid_fps(self) -> None:
        capture = FakeVideoCapture(
            [np.zeros((12, 16, 3), dtype=np.uint8)],
            fps=0.0,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "sample.mp4"
            video_path.write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "invalid FPS"):
                read_video_metadata(video_path, capture_factory=lambda _path: capture)
        self.assertTrue(capture.released)


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


class ManualLabelerDataTests(unittest.TestCase):
    def test_pixel_and_yolo_box_conversion_round_trips(self) -> None:
        label = pixel_box_to_yolo(2, 100, 50, 500, 250, 1000, 500)
        self.assertEqual((label.class_id, label.x_center, label.y_center), (2, 0.3, 0.3))
        self.assertEqual((label.width, label.height), (0.4, 0.4))
        self.assertEqual(
            yolo_to_pixel_box(label, 1000, 500),
            (100, 50, 500, 250),
        )

    def test_pixel_conversion_rejects_out_of_bounds_and_unknown_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "inside the image"):
            pixel_box_to_yolo(0, -1, 0, 10, 10, 100, 100)
        with self.assertRaisesRegex(ValueError, "Unknown class"):
            pixel_box_to_yolo(9, 0, 0, 10, 10, 100, 100)

    def test_annotation_read_write_and_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            label_path = Path(temporary_directory) / "frame.txt"
            labels = [
                pixel_box_to_yolo(0, 10, 20, 50, 80, 100, 100),
                pixel_box_to_yolo(4, 60, 10, 90, 30, 100, 100),
            ]
            write_yolo_annotations(label_path, labels, 100, 100)
            loaded = read_yolo_annotations(label_path, 100, 100)
            self.assertEqual(loaded, labels)

            with self.assertRaises(FileExistsError):
                write_yolo_annotations(label_path, [], 100, 100)
            write_yolo_annotations(
                label_path, [], 100, 100, allow_overwrite=True
            )
            self.assertEqual(label_path.read_text(encoding="utf-8"), "")

    def test_labeling_progress_and_report_track_reviewed_skipped_and_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory) / "dataset"
            create_dataset_structure(dataset_dir)
            images = dataset_dir / "images" / "train"
            labels_dir = dataset_dir / "labels" / "train"
            image_labeled = images / "session_frame_000001_time.png"
            image_empty = images / "session_frame_000002_time.png"
            image_skipped = images / "session_frame_000003_time.png"
            for image_path in (image_labeled, image_empty, image_skipped):
                cv2.imwrite(
                    str(image_path),
                    np.zeros((100, 100, 3), dtype=np.uint8),
                )
            (labels_dir / f"{image_labeled.stem}.txt").write_text(
                "0 0.5 0.5 0.4 0.6\n1 0.2 0.2 0.1 0.1\n",
                encoding="utf-8",
            )
            (labels_dir / f"{image_empty.stem}.txt").write_text("", encoding="utf-8")
            save_labeling_state(
                dataset_dir,
                {"images/train/session_frame_000003_time.png": "skipped"},
            )

            statuses = load_labeling_state(dataset_dir)
            progress = get_labeling_progress(dataset_dir)
            report = build_labeling_report(dataset_dir)

        self.assertEqual(statuses["images/train/session_frame_000003_time.png"], "skipped")
        self.assertEqual(progress.reviewed, 2)
        self.assertEqual(progress.labeled, 1)
        self.assertEqual(progress.skipped, 1)
        self.assertEqual(progress.remaining, 0)
        self.assertEqual(report["images_reviewed"], 2)
        self.assertEqual(report["images_with_labels"], 1)
        self.assertEqual(report["images_without_target_objects"], 1)
        self.assertEqual(report["images_skipped"], 1)
        self.assertEqual(report["total_bounding_boxes"], 2)
        self.assertEqual(report["class_counts"]["player"], 1)
        self.assertEqual(report["class_counts"]["enemy"], 1)


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
