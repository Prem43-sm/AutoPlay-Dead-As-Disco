"""Read-only, purpose-isolated validation/test gameplay capture."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from app.config import CaptureConfig
from capture.frame_metadata import save_sample_frame
from capture.screen_capture import ScreenCapture

Purpose = Literal["validation", "test"]
PURPOSE_TO_SPLIT: dict[Purpose, str] = {"validation": "val", "test": "test"}
DEFAULT_ROOT = Path("data/dataset_independent_eval")
Frame = NDArray[np.uint8]


@dataclass(frozen=True)
class IndependentCaptureConfig:
    capture: CaptureConfig
    purpose: Purpose
    session_id: str
    sample_interval_seconds: float = 0.5
    max_duration_seconds: float = 900.0
    notes: str = ""
    output_root: Path = DEFAULT_ROOT

    def validate(self) -> None:
        self.capture.validate()
        if self.purpose not in PURPOSE_TO_SPLIT:
            raise ValueError("Purpose must be validation or test.")
        if not self.session_id or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", self.session_id
        ):
            raise ValueError(
                "Session ID must be 1-64 letters, digits, underscores, or hyphens "
                "and start with a letter or digit."
            )
        if (
            not isfinite(self.sample_interval_seconds)
            or self.sample_interval_seconds <= 0
        ):
            raise ValueError("Sample interval must be finite and greater than zero.")
        if (
            not isfinite(self.max_duration_seconds)
            or self.max_duration_seconds <= 0
        ):
            raise ValueError("Maximum duration must be finite and greater than zero.")
        if not isinstance(self.notes, str):
            raise ValueError("Session notes must be text.")
        ensure_safe_output_root(self.output_root)


@dataclass(frozen=True)
class IndependentCaptureReport:
    session_id: str
    purpose: Purpose
    split: str
    frame_count: int
    frame_resolution: dict[str, int] | None
    configured_capture_fps: float
    measured_capture_fps: float | None
    capture_interval_seconds: float
    session_manifest: Path
    session_metadata: Path


def ensure_safe_output_root(output_root: Path) -> Path:
    """Prevent any overlap with the legacy or future YOLO datasets."""
    root = Path(output_root).resolve()
    protected_roots = (
        Path("data/dataset").resolve(),
        Path("data/dataset_player_enemy_v2").resolve(),
    )
    for protected in protected_roots:
        if root == protected or root in protected.parents or protected in root.parents:
            raise ValueError(
                f"Independent capture root {root} overlaps protected dataset {protected}."
            )
    return root


def initialize_independent_dataset(output_root: Path = DEFAULT_ROOT) -> Path:
    """Create an isolated YOLO dataset skeleton without overwriting its config."""
    root = ensure_safe_output_root(output_root)
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    config_path = root / "dataset.yaml"
    expected = (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: player\n"
        "  1: enemy\n"
    )
    if config_path.exists():
        if config_path.read_text(encoding="utf-8") != expected:
            raise ValueError(
                f"Refusing to change unexpected independent dataset config: {config_path}"
            )
    else:
        config_path.write_text(expected, encoding="utf-8")
    return root


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(entry, separators=(",", ":")) + "\n")


def record_independent_session(
    config: IndependentCaptureConfig,
    *,
    capture_factory: Any = ScreenCapture,
    clock: Any = time.monotonic,
    sleeper: Any = time.sleep,
    show_preview: bool = True,
) -> IndependentCaptureReport:
    """Capture one complete purpose-bound session; q/Esc or Ctrl+C stops early."""
    config.validate()
    root = initialize_independent_dataset(config.output_root)
    split = PURPOSE_TO_SPLIT[config.purpose]
    session_dir = root / "metadata" / config.purpose / config.session_id
    for purpose in PURPOSE_TO_SPLIT:
        existing_session = root / "metadata" / purpose / config.session_id
        if existing_session.exists():
            raise FileExistsError(f"Independent session already exists: {existing_session}")
    session_dir.mkdir(parents=True)
    image_dir = root / "images" / split
    manifest_path = root / "metadata" / f"{config.purpose}_sessions.jsonl"
    frame_manifest_path = root / "metadata" / "manifest.jsonl"

    started_at = datetime.now(timezone.utc)
    start_clock = clock()
    last_sample_clock = start_clock - config.sample_interval_seconds
    first_capture_clock: float | None = None
    capture_count = 0
    frame_count = 0
    measured_fps: float | None = None
    resolution: dict[str, int] | None = None
    window_name = f"Dead As Disco AI - {config.purpose} capture"
    error: BaseException | None = None

    try:
        with capture_factory(config.capture) as capture:
            while clock() - start_clock < config.max_duration_seconds:
                frame_start = clock()
                frame = capture.capture()
                timestamp = datetime.now(timezone.utc)
                capture_count += 1
                if first_capture_clock is None:
                    first_capture_clock = frame_start
                elif frame_start > first_capture_clock:
                    measured_fps = (capture_count - 1) / (
                        frame_start - first_capture_clock
                    )

                if frame_start - last_sample_clock >= config.sample_interval_seconds:
                    frame_count += 1
                    last_sample_clock = frame_start
                    filename = (
                        f"{config.session_id}_frame_{frame_count:06d}_"
                        f"{timestamp.strftime('%Y%m%dT%H%M%S_%fZ')}.png"
                    )
                    image_path = image_dir / filename
                    saved_image, sidecar_path = save_sample_frame(
                        frame,
                        image_path,
                        frame_number=frame_count,
                        capture_fps=measured_fps,
                        monitor_index=config.capture.monitor_index,
                        monitor_bounds=capture.monitor_bounds,
                        capture_region=capture.bounds,
                        timestamp=timestamp,
                    )
                    resolution = {
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                    }
                    relative_image = saved_image.relative_to(root).as_posix()
                    relative_sidecar = sidecar_path.relative_to(root).as_posix()
                    sidecar_data = json.loads(
                        sidecar_path.read_text(encoding="utf-8")
                    )
                    sidecar_data.update(
                        {
                            "source": "live_screen_capture",
                            "source_video": None,
                            "source_session": config.session_id,
                            "purpose": config.purpose,
                            "split": split,
                            "image_path": relative_image,
                            "metadata_path": relative_sidecar,
                            "labels": [],
                            "labeled": False,
                        }
                    )
                    sidecar_path.write_text(
                        json.dumps(sidecar_data, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    _append_jsonl(
                        frame_manifest_path,
                        {
                            **sidecar_data,
                            "session_notes": config.notes,
                        },
                    )

                if show_preview:
                    preview = frame.copy()
                    cv2.putText(
                        preview,
                        f"{config.purpose} | {frame_count} frames | "
                        f"{measured_fps:.1f} FPS"
                        if measured_fps is not None
                        else f"{config.purpose} | {frame_count} frames",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, preview)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

                remaining_duration = config.max_duration_seconds - (
                    clock() - start_clock
                )
                remaining_sample = config.sample_interval_seconds - (
                    clock() - last_sample_clock
                )
                sleep_for = min(
                    max(0.0, remaining_sample),
                    max(0.0, remaining_duration),
                    1.0 / config.capture.max_fps,
                )
                if sleep_for > 0:
                    sleeper(sleep_for)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        error = exc
        raise
    finally:
        if show_preview:
            cv2.destroyAllWindows()
        ended_at = datetime.now(timezone.utc)
        session_data: dict[str, Any] = {
            "session_id": config.session_id,
            "purpose": config.purpose,
            "source": "live_screen_capture",
            "source_video": None,
            "split": split,
            "source_resolution": resolution,
            "capture_fps": {
                "configured_max": config.capture.max_fps,
                "measured_average": measured_fps,
            },
            "capture_interval_seconds": config.sample_interval_seconds,
            "frame_count": frame_count,
            "capture_count": capture_count,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "timestamp": started_at.isoformat(),
            "monitor_index": config.capture.monitor_index,
            "monitor_bounds": capture.monitor_bounds if capture_count else None,
            "capture_region": capture.bounds if capture_count else None,
            "max_duration_seconds": config.max_duration_seconds,
            "notes": config.notes,
            "labels_generated": False,
            "complete": error is None,
        }
        session_metadata = session_dir / "session.json"
        session_metadata.write_text(
            json.dumps(session_data, indent=2) + "\n",
            encoding="utf-8",
        )
        _append_jsonl(manifest_path, session_data)

    return IndependentCaptureReport(
        session_id=config.session_id,
        purpose=config.purpose,
        split=split,
        frame_count=frame_count,
        frame_resolution=resolution,
        configured_capture_fps=config.capture.max_fps,
        measured_capture_fps=measured_fps,
        capture_interval_seconds=config.sample_interval_seconds,
        session_manifest=manifest_path,
        session_metadata=session_metadata,
    )
