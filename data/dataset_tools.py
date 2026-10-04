import json
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from app.config import CaptureConfig
from app.logger import get_logger
from capture.frame_metadata import save_sample_frame
from capture.screen_capture import ScreenCapture
from vision.models import BoundingBox

DatasetSplit = Literal["train", "val", "test"]
CLASS_NAMES = (
    "player",
    "enemy",
    "stunned_enemy",
    "airborne_enemy",
    "action_prompt",
    "takedown_indicator",
)
SPLITS: tuple[DatasetSplit, ...] = ("train", "val", "test")
YOLO_CLASS_NAMES = {index: name for index, name in enumerate(CLASS_NAMES)}
Frame = NDArray[np.uint8]


@dataclass(frozen=True)
class DatasetCaptureConfig:
    capture: CaptureConfig
    sample_interval: float
    max_samples: int
    output_dir: Path
    session_id: str | None = None
    split: DatasetSplit | None = None
    deduplicate: bool = False
    similarity_threshold: float = 2.0

    def validate(self) -> None:
        self.capture.validate()
        if not isfinite(self.sample_interval) or self.sample_interval <= 0:
            raise ValueError("Sample interval must be greater than zero.")
        if self.max_samples < 1:
            raise ValueError("Maximum samples must be at least 1.")
        if self.split is not None and self.split not in SPLITS:
            raise ValueError(f"Split must be one of: {', '.join(SPLITS)}.")
        if (
            not np.isfinite(self.similarity_threshold)
            or self.similarity_threshold < 0
        ):
            raise ValueError("Similarity threshold must be finite and non-negative.")


@dataclass(frozen=True)
class DatasetCaptureReport:
    session_id: str
    split: DatasetSplit
    captured_frames: int
    sampled_frames: int
    saved_frames: int
    duplicate_frames_skipped: int
    manifest_path: Path
    session_metadata_path: Path


@dataclass(frozen=True)
class YoloLabel:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_bbox(self, image_width: int, image_height: int) -> BoundingBox:
        width = max(1, round(self.width * image_width))
        height = max(1, round(self.height * image_height))
        x = round(self.x_center * image_width - width / 2)
        y = round(self.y_center * image_height - height / 2)
        return BoundingBox(x, y, width, height)


@dataclass(frozen=True)
class DatasetValidationReport:
    split_image_counts: dict[str, int]
    total_image_count: int
    labeled_image_count: int
    unlabeled_image_count: int
    multi_class_image_count: int
    no_target_class_image_count: int
    exact_duplicate_image_count: int
    near_duplicate_pair_count: int
    invalid_label_count: int
    label_object_counts: dict[str, int]
    class_frame_counts: dict[str, int]
    session_split_leaks: tuple[str, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        report = asdict(self)
        report["valid"] = self.valid
        return report


def create_dataset_structure(dataset_dir: Path) -> None:
    dataset_dir = Path(dataset_dir)
    for split in SPLITS:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (dataset_dir / "metadata").mkdir(parents=True, exist_ok=True)

    yaml_path = dataset_dir / "dataset.yaml"
    if not yaml_path.exists():
        names = "\n".join(f"  {index}: {name}" for index, name in YOLO_CLASS_NAMES.items())
        yaml_path.write_text(
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            f"names:\n{names}\n",
            encoding="utf-8",
        )


def assign_session_split(session_id: str) -> DatasetSplit:
    bucket = int.from_bytes(sha256(session_id.encode("utf-8")).digest()[:4], "big") % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "val"
    return "test"


def frame_filename(frame_number: int, timestamp: datetime) -> str:
    if frame_number < 1:
        raise ValueError("Frame number must be at least 1.")
    if timestamp.tzinfo is None:
        raise ValueError("Frame timestamp must include timezone information.")
    return f"frame_{frame_number:06d}_{timestamp.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}.png"


def frame_difference(previous: Frame, current: Frame) -> float:
    if previous.shape != current.shape:
        return float("inf")
    previous_small = cv2.resize(previous, (64, 36), interpolation=cv2.INTER_AREA)
    current_small = cv2.resize(current, (64, 36), interpolation=cv2.INTER_AREA)
    previous_gray = cv2.cvtColor(previous_small, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)
    return float(
        cv2.absdiff(previous_gray, current_gray).mean()
    )


def is_duplicate_frame(
    previous: Frame | None, current: Frame, threshold: float
) -> bool:
    if previous is None:
        return False
    if threshold < 0 or not np.isfinite(threshold):
        raise ValueError("Similarity threshold must be finite and non-negative.")
    return frame_difference(previous, current) <= threshold


def _normalize_session_id(session_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", session_id).strip("_")
    if not normalized:
        raise ValueError("Session ID must contain at least one letter or digit.")
    return normalized[:64]


def record_dataset_session(
    config: DatasetCaptureConfig,
    *,
    capture_factory: Any = ScreenCapture,
    clock: Any = time.monotonic,
    sleeper: Any = time.sleep,
) -> DatasetCaptureReport:
    config.validate()
    dataset_dir = Path(config.output_dir)
    create_dataset_structure(dataset_dir)
    logger = get_logger(__name__)
    session_id = _normalize_session_id(
        config.session_id
        or datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%S_%fZ")
    )
    split = config.split or assign_session_split(session_id)
    image_dir = dataset_dir / "images" / split
    metadata_dir = dataset_dir / "metadata" / session_id
    session_metadata_path = metadata_dir / "session.json"
    manifest_path = dataset_dir / "metadata" / "manifest.jsonl"
    sessions_dir = dataset_dir.parent / "sessions"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_record_path = sessions_dir / f"{session_id}.json"
    if session_metadata_path.exists() or session_record_path.exists():
        raise FileExistsError(f"Dataset session already exists: {session_metadata_path}")

    start_wall = datetime.now(timezone.utc)
    start_clock = clock()
    capture_count = 0
    sampled_count = 0
    saved_count = 0
    duplicates_skipped = 0
    previous_saved: Frame | None = None
    last_sample_clock = start_clock - config.sample_interval
    first_frame_clock: float | None = None
    latest_capture_fps: float | None = None
    try:
        with capture_factory(config.capture) as capture:
            logger.info(
                "Dataset session %s: split=%s monitor=%d region=%s, "
                "sample interval=%.2fs, max samples=%d.",
                session_id,
                split,
                config.capture.monitor_index,
                capture.bounds,
                config.sample_interval,
                config.max_samples,
            )
            while sampled_count < config.max_samples:
                frame = capture.capture()
                now_clock = clock()
                timestamp = datetime.now(timezone.utc)
                capture_count += 1
                if first_frame_clock is None:
                    first_frame_clock = now_clock
                elif now_clock > first_frame_clock:
                    latest_capture_fps = (capture_count - 1) / (
                        now_clock - first_frame_clock
                    )

                if now_clock - last_sample_clock >= config.sample_interval:
                    sampled_count += 1
                    last_sample_clock = now_clock
                    if config.deduplicate and is_duplicate_frame(
                        previous_saved, frame, config.similarity_threshold
                    ):
                        duplicates_skipped += 1
                        logger.info(
                            "Skipped sampled frame %d as visually similar "
                            "(threshold %.3f).",
                            sampled_count,
                            config.similarity_threshold,
                        )
                    else:
                        filename = frame_filename(sampled_count, timestamp)
                        image_path = image_dir / f"{session_id}_{filename}"
                        metadata_image_path = metadata_dir / f"{image_path.name}"
                        _, sidecar_path = save_sample_frame(
                            frame,
                            metadata_image_path,
                            frame_number=sampled_count,
                            capture_fps=latest_capture_fps,
                            monitor_index=config.capture.monitor_index,
                            monitor_bounds=capture.monitor_bounds,
                            capture_region=capture.bounds,
                            timestamp=timestamp,
                        )
                        image_metadata = json.loads(
                            sidecar_path.read_text(encoding="utf-8")
                        )
                        dataset_metadata = {
                            **image_metadata,
                            "image_path": image_path.relative_to(dataset_dir).as_posix(),
                            "source_session": session_id,
                            "split": split,
                            "labels": [],
                            "labeled": False,
                        }
                        sidecar_path.write_text(
                            json.dumps(dataset_metadata, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        shutil.move(str(metadata_image_path), image_path)
                        with manifest_path.open(
                            "a", encoding="utf-8", newline="\n"
                        ) as manifest:
                            manifest.write(
                                json.dumps(dataset_metadata, separators=(",", ":"))
                                + "\n"
                            )
                        previous_saved = frame.copy()
                        saved_count += 1
                        logger.info(
                            "Saved dataset frame %d: %s (metadata: %s).",
                            sampled_count,
                            image_path,
                            sidecar_path,
                        )
                if sampled_count < config.max_samples:
                    remaining = config.sample_interval - (
                        clock() - last_sample_clock
                    )
                    if remaining > 0:
                        sleeper(min(remaining, 1 / config.capture.max_fps))

        end_wall = datetime.now(timezone.utc)
        session_data = {
            "session_id": session_id,
            "split": split,
            "started_at": start_wall.isoformat(),
            "ended_at": end_wall.isoformat(),
            "monitor_index": config.capture.monitor_index,
            "capture_region": (
                asdict(config.capture.region)
                if config.capture.region is not None
                else None
            ),
            "capture_fps_limit": config.capture.max_fps,
            "sample_interval_seconds": config.sample_interval,
            "max_samples": config.max_samples,
            "captured_frames": capture_count,
            "sampled_frames": sampled_count,
            "saved_frames": saved_count,
            "duplicate_frames_skipped": duplicates_skipped,
            "labels_generated": False,
        }
        session_metadata_path.write_text(
            json.dumps(session_data, indent=2) + "\n",
            encoding="utf-8",
        )
        session_record_path.write_text(
            json.dumps(session_data, indent=2) + "\n",
            encoding="utf-8",
        )
        return DatasetCaptureReport(
            session_id=session_id,
            split=split,
            captured_frames=capture_count,
            sampled_frames=sampled_count,
            saved_frames=saved_count,
            duplicate_frames_skipped=duplicates_skipped,
            manifest_path=manifest_path,
            session_metadata_path=session_metadata_path,
        )
    except Exception:
        logger.exception(
            "Dataset recording failed for session %s after %d saved frames.",
            session_id,
            saved_count,
        )
        raise


def parse_yolo_labels(label_path: Path) -> list[YoloLabel]:
    labels: list[YoloLabel] = []
    for line_number, raw_line in enumerate(
        Path(label_path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path}:{line_number}: expected 5 fields, got {len(parts)}."
            )
        try:
            class_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(
                f"{label_path}:{line_number}: class ID must be an integer "
                "and coordinates must be numeric."
            ) from exc
        labels.append(YoloLabel(class_id, *coordinates))
    return labels


def validate_yolo_label(
    label: YoloLabel, image_width: int, image_height: int, class_names: dict[int, str]
) -> str | None:
    if label.class_id not in class_names:
        return f"unknown class ID {label.class_id}"
    values = (label.x_center, label.y_center, label.width, label.height)
    if not all(np.isfinite(value) for value in values):
        return "coordinates must be finite"
    if label.width <= 0 or label.height <= 0:
        return "box width and height must be positive"
    if not (0 <= label.x_center <= 1 and 0 <= label.y_center <= 1):
        return "box center must be normalized to [0, 1]"
    if label.width > 1 or label.height > 1:
        return "box width and height must be normalized to (0, 1]"
    if (
        label.x_center - label.width / 2 < 0
        or label.y_center - label.height / 2 < 0
        or label.x_center + label.width / 2 > 1
        or label.y_center + label.height / 2 > 1
    ):
        return "bounding box extends outside image boundaries"
    if image_width <= 0 or image_height <= 0:
        return "image dimensions must be positive"
    return None


def load_class_names(dataset_dir: Path) -> dict[int, str]:
    yaml_path = Path(dataset_dir) / "dataset.yaml"
    if not yaml_path.is_file():
        return YOLO_CLASS_NAMES.copy()
    names: dict[int, str] = {}
    in_names = False
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "names:":
            in_names = True
            continue
        if not in_names or ":" not in line:
            continue
        class_part, name = line.split(":", 1)
        if class_part.strip().isdigit():
            names[int(class_part.strip())] = name.strip()
    if not names:
        raise ValueError(f"No class names found in {yaml_path}.")
    return names


def _capture_sequence(filename: str) -> tuple[str, int] | None:
    match = re.match(r"^(?P<session>.+)_frame_(?P<frame>\d{6})_", filename)
    if match is None:
        return None
    return match.group("session"), int(match.group("frame"))


def validate_dataset(dataset_dir: Path) -> DatasetValidationReport:
    dataset_dir = Path(dataset_dir)
    dataset_root = dataset_dir.resolve()
    class_names = load_class_names(dataset_dir)
    errors: list[str] = []
    warnings: list[str] = []
    counts = {split: 0 for split in SPLITS}
    label_counts = {name: 0 for name in class_names.values()}
    class_frame_counts = {name: 0 for name in class_names.values()}
    seen_filenames: dict[str, Path] = {}
    labeled_images = 0
    unlabeled_images = 0
    multi_class_images = 0
    no_target_images = 0
    invalid_label_count = 0
    exact_duplicate_images = 0
    content_hashes: set[str] = set()
    sequence_images: list[tuple[str, int, Path]] = []
    session_splits: dict[str, set[str]] = {}
    manifest_sessions: dict[str, str] = {}

    manifest_path = dataset_dir / "metadata" / "manifest.jsonl"
    if manifest_path.exists():
        for line_number, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                manifest_entry = json.loads(line)
                image_path = (dataset_dir / manifest_entry["image_path"]).resolve()
                image_path.relative_to(dataset_root)
                image_key = image_path.relative_to(dataset_root).as_posix()
                source_session = manifest_entry["source_session"]
                if not isinstance(source_session, str) or not source_session:
                    raise ValueError("source_session must be a non-empty string")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                errors.append(
                    f"{manifest_path}:{line_number}: invalid manifest entry: {exc}"
                )
                continue
            manifest_sessions[image_key] = source_session
            if not image_path.is_file():
                errors.append(
                    f"{manifest_path}:{line_number}: missing image {image_path}"
                )

    for split in SPLITS:
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if image_dir.exists() and path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        ) if image_dir.exists() else []
        counts[split] = len(image_paths)
        for image_path in image_paths:
            previous_path = seen_filenames.get(image_path.name.casefold())
            if previous_path is not None:
                errors.append(
                    f"Duplicate image filename {image_path.name}: {previous_path} and {image_path}"
                )
            else:
                seen_filenames[image_path.name.casefold()] = image_path

            image_key = image_path.resolve().relative_to(dataset_root).as_posix()
            sequence = _capture_sequence(image_path.name)
            source_session = manifest_sessions.get(
                image_key, sequence[0] if sequence is not None else None
            )
            if source_session is not None:
                session_splits.setdefault(source_session, set()).add(split)
            if sequence is not None:
                sequence_images.append((sequence[0], sequence[1], image_path))

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                errors.append(f"Corrupted or unreadable image: {image_path}")
                continue
            try:
                content_hash = sha256(image_path.read_bytes()).hexdigest()
            except OSError as exc:
                errors.append(f"Could not hash image {image_path}: {exc}")
            else:
                if content_hash in content_hashes:
                    exact_duplicate_images += 1
                else:
                    content_hashes.add(content_hash)

            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                warnings.append(f"Missing label (unlabeled image): {label_path}")
                unlabeled_images += 1
                continue
            labeled_images += 1
            try:
                labels = parse_yolo_labels(label_path)
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
                invalid_label_count += 1
                continue
            image_classes: set[str] = set()
            for label in labels:
                problem = validate_yolo_label(
                    label, image.shape[1], image.shape[0], class_names
                )
                if problem:
                    errors.append(f"{label_path}: {problem}")
                    invalid_label_count += 1
                else:
                    label_counts[class_names[label.class_id]] += 1
                    image_classes.add(class_names[label.class_id])
            for class_name in image_classes:
                class_frame_counts[class_name] += 1
            if len(image_classes) > 1:
                multi_class_images += 1
            if not labels:
                no_target_images += 1

        if label_dir.exists():
            image_stems = {path.stem for path in image_paths}
            for label_path in label_dir.glob("*.txt"):
                if label_path.stem not in image_stems:
                    errors.append(f"Label has no matching image: {label_path}")

    for session_id, splits in sorted(session_splits.items()):
        if len(splits) > 1:
            errors.append(
                f"Gameplay session {session_id} is split across: {', '.join(sorted(splits))}."
            )

    near_duplicate_pairs = 0
    ordered_sequences = sorted(sequence_images, key=lambda item: (item[0], item[1]))
    for previous, current in zip(ordered_sequences, ordered_sequences[1:]):
        previous_session, previous_number, previous_path = previous
        current_session, current_number, current_path = current
        if previous_session != current_session or current_number != previous_number + 1:
            continue
        previous_image = cv2.imread(str(previous_path), cv2.IMREAD_COLOR)
        current_image = cv2.imread(str(current_path), cv2.IMREAD_COLOR)
        if previous_image is None or current_image is None:
            continue
        if is_duplicate_frame(previous_image, current_image, 2.0):
            near_duplicate_pairs += 1

    total_images = sum(counts.values())
    if total_images == 0:
        warnings.append("Dataset contains no images.")
    return DatasetValidationReport(
        split_image_counts=counts,
        total_image_count=total_images,
        labeled_image_count=labeled_images,
        unlabeled_image_count=unlabeled_images,
        multi_class_image_count=multi_class_images,
        no_target_class_image_count=no_target_images,
        exact_duplicate_image_count=exact_duplicate_images,
        near_duplicate_pair_count=near_duplicate_pairs,
        invalid_label_count=invalid_label_count,
        label_object_counts=label_counts,
        class_frame_counts=class_frame_counts,
        session_split_leaks=tuple(
            session_id
            for session_id, splits in sorted(session_splits.items())
            if len(splits) > 1
        ),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def render_dataset_preview(
    image_path: Path, label_path: Path | None = None, dataset_dir: Path | None = None
) -> Frame:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read preview image: {image_path}")
    if label_path is None:
        if dataset_dir is None:
            raise ValueError("Provide either label_path or dataset_dir.")
        image_path = Path(image_path)
        split = image_path.parent.name
        if split not in SPLITS:
            raise ValueError("Image must be inside an images/{train,val,test} directory.")
        label_path = Path(dataset_dir) / "labels" / split / f"{image_path.stem}.txt"
    if not Path(label_path).is_file():
        raise FileNotFoundError(f"No manual label file found: {label_path}")

    class_names = load_class_names(dataset_dir) if dataset_dir else YOLO_CLASS_NAMES
    height, width = image.shape[:2]
    for label in parse_yolo_labels(label_path):
        problem = validate_yolo_label(label, width, height, class_names)
        if problem:
            raise ValueError(f"{label_path}: {problem}")
        box = label.to_bbox(width, height).clip(width, height)
        if box is None:
            raise ValueError(f"{label_path}: label produced an empty box.")
        cv2.rectangle(
            image, (box.x, box.y), (box.x + box.width, box.y + box.height), (0, 255, 0), 2
        )
        cv2.putText(
            image,
            class_names[label.class_id],
            (box.x, max(18, box.y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return image
