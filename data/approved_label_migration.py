"""Copy explicitly approved player/enemy boxes into the v2 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

from data.dataset_tools import (
    load_class_names,
    parse_yolo_labels,
    validate_yolo_label,
    YoloLabel,
)

ALLOWED_CLASS_NAMES = {"player", "enemy"}
SPLITS = {"train", "val", "test"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PlannedCopy:
    image_relative_path: str
    label_relative_path: str
    session_id: str
    target_split: str
    image_bytes: bytes
    label_bytes: bytes
    box_count: int


def _safe_relative_path(value: Any, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a non-empty POSIX relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must not escape the dataset root.")
    return path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_session_index(source_root: Path) -> dict[str, str]:
    manifest_path = source_root / "metadata" / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Legacy manifest not found: {manifest_path}")
    sessions: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            image_path = _safe_relative_path(entry["image_path"], "manifest image_path")
            session_id = entry["source_session"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{manifest_path}:{line_number}: invalid manifest entry: {exc}"
            ) from exc
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"{manifest_path}:{line_number}: invalid source_session.")
        key = image_path.as_posix()
        previous = sessions.get(key)
        if previous is not None and previous != session_id:
            raise ValueError(
                f"{manifest_path}:{line_number}: conflicting sessions for {key}."
            )
        sessions[key] = session_id
    return sessions


def prepare_migration_plan(
    source_root: Path,
    destination_root: Path,
    approvals_path: Path,
) -> list[PlannedCopy]:
    """Validate explicit human approvals and return a read-only copy plan."""
    source_root = Path(source_root).resolve()
    destination_root = Path(destination_root).resolve()
    approvals_data = json.loads(Path(approvals_path).read_text(encoding="utf-8"))
    if (
        not isinstance(approvals_data, dict)
        or type(approvals_data.get("schema_version")) is not int
        or approvals_data.get("schema_version") != 1
    ):
        raise ValueError("Approval manifest schema_version must be 1.")
    approvals = approvals_data.get("approvals")
    if not isinstance(approvals, list):
        raise ValueError("Approval manifest must contain an approvals list.")

    class_names = load_class_names(source_root)
    destination_config = destination_root / "dataset.yaml"
    if not destination_config.is_file():
        raise FileNotFoundError(
            f"Versioned destination config not found: {destination_config}"
        )
    if load_class_names(destination_root) != {0: "player", 1: "enemy"}:
        raise ValueError(
            "Destination dataset.yaml must define exactly class 0 player and class 1 enemy."
        )
    session_index = _load_session_index(source_root)
    session_splits: dict[str, str] = {}
    seen_images: set[str] = set()
    seen_outputs: set[str] = set()
    plan: list[PlannedCopy] = []

    for approval_index, approval in enumerate(approvals, start=1):
        prefix = f"approvals[{approval_index}]"
        if not isinstance(approval, dict):
            raise ValueError(f"{prefix} must be an object.")
        reviewer = approval.get("approved_by")
        reviewed_at = approval.get("reviewed_at")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError(f"{prefix}.approved_by is required.")
        if not isinstance(reviewed_at, str) or not reviewed_at.strip():
            raise ValueError(f"{prefix}.reviewed_at is required.")

        image_rel = _safe_relative_path(approval.get("source_image"), f"{prefix}.source_image")
        label_rel = _safe_relative_path(approval.get("source_label"), f"{prefix}.source_label")
        image_key = image_rel.as_posix()
        if image_key in seen_images:
            raise ValueError(f"{prefix}: duplicate source image {image_key}.")
        seen_images.add(image_key)
        if (
            len(image_rel.parts) != 3
            or image_rel.parts[0] != "images"
            or image_rel.parts[1] not in SPLITS
            or image_rel.suffix.lower() != ".png"
        ):
            raise ValueError(f"{prefix}.source_image must identify a legacy split PNG.")
        expected_label = PurePosixPath("labels", image_rel.parts[1], image_rel.stem + ".txt")
        if label_rel != expected_label:
            raise ValueError(f"{prefix}.source_label must match the source image.")

        target_split = approval.get("target_split")
        if not isinstance(target_split, str) or target_split not in SPLITS:
            raise ValueError(f"{prefix}.target_split must be train, val, or test.")
        session_id = session_index.get(image_key)
        if session_id is None:
            raise ValueError(f"{prefix}: source image is absent from the legacy manifest.")
        assigned_split = session_splits.setdefault(session_id, target_split)
        if assigned_split != target_split:
            raise ValueError(
                f"Session {session_id!r} is assigned to multiple destination splits."
            )

        image_source = (source_root / Path(*image_rel.parts)).resolve()
        label_source = (source_root / Path(*label_rel.parts)).resolve()
        try:
            image_source.relative_to(source_root)
            label_source.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"{prefix}: source path escapes the legacy dataset.") from exc
        image_bytes = image_source.read_bytes()
        label_bytes = label_source.read_bytes()
        expected_image_hash = approval.get("image_sha256")
        expected_label_hash = approval.get("label_sha256")
        if not isinstance(expected_image_hash, str) or not SHA256_PATTERN.fullmatch(
            expected_image_hash
        ):
            raise ValueError(f"{prefix}.image_sha256 must be a lowercase SHA-256.")
        if not isinstance(expected_label_hash, str) or not SHA256_PATTERN.fullmatch(
            expected_label_hash
        ):
            raise ValueError(f"{prefix}.label_sha256 must be a lowercase SHA-256.")
        if _sha256(image_bytes) != expected_image_hash:
            raise ValueError(f"{prefix}: source image checksum changed.")
        if _sha256(label_bytes) != expected_label_hash:
            raise ValueError(f"{prefix}: source label checksum changed.")

        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"{prefix}: source image cannot be decoded.")
        height, width = frame.shape[:2]
        parsed = parse_yolo_labels(label_source)
        raw_lines = label_bytes.decode("utf-8").splitlines()
        parsed_by_line: dict[int, YoloLabel] = {}
        parsed_index = 0
        for line_number, raw_line in enumerate(raw_lines, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parsed_by_line[line_number] = parsed[parsed_index]
            parsed_index += 1

        boxes = approval.get("approved_boxes")
        if not isinstance(boxes, list) or not boxes:
            raise ValueError(f"{prefix}.approved_boxes must contain explicit box approvals.")
        approved_line_numbers: set[int] = set()
        output_lines: list[str] = []
        for box_index, box in enumerate(boxes, start=1):
            box_prefix = f"{prefix}.approved_boxes[{box_index}]"
            if not isinstance(box, dict):
                raise ValueError(f"{box_prefix} must be an object.")
            line_number = box.get("line_number")
            class_name = box.get("class_name")
            if not isinstance(line_number, int) or isinstance(line_number, bool):
                raise ValueError(f"{box_prefix}.line_number must be a 1-based integer.")
            if line_number in approved_line_numbers:
                raise ValueError(f"{box_prefix}: duplicate approved line {line_number}.")
            approved_line_numbers.add(line_number)
            if (
                not isinstance(class_name, str)
                or class_name not in ALLOWED_CLASS_NAMES
            ):
                raise ValueError(
                    f"{box_prefix}.class_name must be exactly player or enemy; "
                    "state/UI classes are never converted."
                )
            label = parsed_by_line.get(line_number)
            if label is None:
                raise ValueError(f"{box_prefix}: line {line_number} is not a YOLO box.")
            if class_names.get(label.class_id) != class_name:
                raise ValueError(
                    f"{box_prefix}: approved class does not match the legacy class."
                )
            invalid_reason = validate_yolo_label(label, width, height, class_names)
            if invalid_reason is not None:
                raise ValueError(
                    f"{box_prefix}: invalid legacy box ({invalid_reason}); "
                    "review and correct it manually before approval."
                )
            original_line = raw_lines[line_number - 1].strip()
            output_lines.append(original_line)

        output_image_rel = PurePosixPath("images", target_split, image_rel.name).as_posix()
        output_label_rel = PurePosixPath(
            "labels", target_split, image_rel.stem + ".txt"
        ).as_posix()
        if output_image_rel in seen_outputs or output_label_rel in seen_outputs:
            raise ValueError(f"{prefix}: duplicate destination for {image_rel.name}.")
        seen_outputs.update((output_image_rel, output_label_rel))
        plan.append(
            PlannedCopy(
                image_relative_path=output_image_rel,
                label_relative_path=output_label_rel,
                session_id=session_id,
                target_split=target_split,
                image_bytes=image_bytes,
                label_bytes=("\n".join(output_lines) + "\n").encode("utf-8"),
                box_count=len(output_lines),
            )
        )

    for item in plan:
        for relative_path in (item.image_relative_path, item.label_relative_path):
            destination = (destination_root / Path(*PurePosixPath(relative_path).parts)).resolve()
            try:
                destination.relative_to(destination_root)
            except ValueError as exc:
                raise ValueError("Destination path escapes the v2 dataset.") from exc
            if destination.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing destination: {destination}"
                )
    return plan


def apply_migration_plan(destination_root: Path, plan: list[PlannedCopy]) -> None:
    """Write a fully prevalidated plan without replacing any existing files."""
    destination_root = Path(destination_root).resolve()
    outputs: list[tuple[Path, bytes]] = []
    for item in plan:
        image_rel = _safe_relative_path(item.image_relative_path, "planned image path")
        label_rel = _safe_relative_path(item.label_relative_path, "planned label path")
        image_path = (destination_root / Path(*image_rel.parts)).resolve()
        label_path = (destination_root / Path(*label_rel.parts)).resolve()
        for output_path in (image_path, label_path):
            try:
                output_path.relative_to(destination_root)
            except ValueError as exc:
                raise ValueError("Destination path escapes the v2 dataset.") from exc
            if output_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing destination: {output_path}"
                )
        outputs.extend(
            (
                (image_path, item.image_bytes),
                (label_path, item.label_bytes),
            )
        )

    for output_path, contents in outputs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(contents)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run or apply explicit player/enemy box approvals to dataset v2."
    )
    parser.add_argument("--source", type=Path, default=Path("data/dataset"))
    parser.add_argument(
        "--destination", type=Path, default=Path("data/dataset_player_enemy_v2")
    )
    parser.add_argument(
        "--approvals",
        type=Path,
        required=True,
        help="Reviewer-completed JSON approval manifest.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Copy the validated plan. Without this flag, only a dry-run is performed.",
    )
    args = parser.parse_args()
    plan = prepare_migration_plan(args.source, args.destination, args.approvals)
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry-run",
                "approved_images": len(plan),
                "approved_boxes": sum(item.box_count for item in plan),
                "sessions": {
                    item.session_id: item.target_split
                    for item in plan
                },
                "destinations": [
                    {
                        "image": item.image_relative_path,
                        "label": item.label_relative_path,
                        "boxes": item.box_count,
                    }
                    for item in plan
                ],
            },
            indent=2,
        )
    )
    if args.apply:
        apply_migration_plan(args.destination, plan)


if __name__ == "__main__":
    main()
