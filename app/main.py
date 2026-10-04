import argparse
import json
from pathlib import Path

import cv2

from app.config import CaptureConfig, CaptureRegion
from app.logger import get_logger
from capture.screen_capture import run_live_capture
from data.dataset_tools import (
    DatasetCaptureConfig,
    VideoDatasetCaptureConfig,
    create_dataset_structure,
    extract_video_dataset_session,
    record_dataset_session,
    render_dataset_preview,
    validate_dataset,
)
from data.dataset_labeler import build_labeling_report, run_dataset_labeler
from vision.state_estimator import (
    draw_debug_frame,
    estimate_game_state,
    load_frame_context,
    save_vision_debug,
)
from vision.detector import VisionDetector
from vision.prompt_detector import PromptDetector, pytesseract_reader


def _parse_region(value: str) -> CaptureRegion:
    try:
        left, top, width, height = (int(part) for part in value.split(","))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "Region must be four comma-separated integers: left,top,width,height."
        ) from exc

    region = CaptureRegion(left, top, width, height)
    try:
        region.validate()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return region


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display and measure live monitor screen capture."
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="1-based monitor index (default: 1).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Maximum capture rate (default: 60).",
    )
    parser.add_argument(
        "--region",
        type=_parse_region,
        help="Optional monitor-relative region: left,top,width,height.",
    )
    parser.add_argument(
        "--save-frame",
        type=Path,
        help="Save the first frame to this image path, with a JSON sidecar.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("recordings/samples"),
        help="Directory for sample frames (default: recordings/samples).",
    )
    parser.add_argument(
        "--save-first-frame",
        action="store_true",
        help="Save the first frame and JSON metadata in --output-dir.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        help="Stop automatically after this many frames.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Capture without opening the preview window.",
    )
    parser.add_argument(
        "--vision-image",
        type=Path,
        help="Analyze a saved image and print its preliminary GameState as JSON.",
    )
    parser.add_argument(
        "--vision-debug",
        action="store_true",
        help="Display an annotated preview; requires --vision-image.",
    )
    parser.add_argument(
        "--vision-ocr",
        action="store_true",
        help="Use optional pytesseract OCR to read E/F prompt labels.",
    )
    parser.add_argument(
        "--save-vision-debug",
        action="store_true",
        help="Save one annotated frame and GameState JSON; requires --vision-image.",
    )
    parser.add_argument(
        "--vision-output-dir",
        type=Path,
        default=Path("recordings/vision"),
        help="Output directory for one saved vision debug frame.",
    )
    parser.add_argument(
        "--dataset-init",
        type=Path,
        metavar="DATASET_DIR",
        help="Create the dataset image/label/metadata directory structure.",
    )
    parser.add_argument(
        "--dataset-capture",
        action="store_true",
        help="Record a bounded, read-only gameplay dataset session.",
    )
    parser.add_argument(
        "--video-dataset-capture",
        type=Path,
        metavar="VIDEO",
        help="Extract bounded frames from a video file into the dataset.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/dataset"),
        help="Dataset root directory (default: data/dataset).",
    )
    parser.add_argument(
        "--dataset-session",
        help="Optional session ID; defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--dataset-split",
        choices=("train", "val", "test", "auto"),
        default="auto",
        help="Assign the entire session to a split (default: deterministic auto).",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Seconds between sampled frames (default: 1).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum sampled frames per session, including dedup skips (default: 100).",
    )
    parser.add_argument(
        "--video-sample-interval",
        type=float,
        default=0.5,
        help="Seconds between video samples (default: 0.5, approximately 2 FPS).",
    )
    parser.add_argument(
        "--video-max-frames",
        type=int,
        default=1000,
        help="Maximum video sample opportunities, including dedup skips (default: 1000).",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Skip sampled frames with little visual change from the last saved frame.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=2.0,
        help="Mean grayscale pixel difference threshold from 0 to 255 (default: 2).",
    )
    parser.add_argument(
        "--dataset-validate",
        type=Path,
        metavar="DATASET_DIR",
        help="Validate images and manual YOLO labels; prints a JSON report.",
    )
    parser.add_argument(
        "--dataset-preview",
        type=Path,
        metavar="IMAGE",
        help="Preview a dataset image with its manual labels only.",
    )
    parser.add_argument(
        "--dataset-label",
        type=Path,
        help="Optional YOLO label file for --dataset-preview.",
    )
    parser.add_argument(
        "--dataset-labeler",
        action="store_true",
        help="Open the local manual bounding-box labeling tool.",
    )
    parser.add_argument(
        "--labeler-split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="Filter images shown by --dataset-labeler (default: all).",
    )
    parser.add_argument(
        "--labeler-session",
        help="Filter --dataset-labeler to one capture session ID.",
    )
    parser.add_argument(
        "--dataset-label-report",
        action="store_true",
        help="Print manual-labeling progress and per-class counts as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_operations = (
        args.dataset_init is not None,
        args.dataset_capture,
        args.video_dataset_capture is not None,
        args.dataset_validate is not None,
        args.dataset_preview is not None,
        args.dataset_labeler,
        args.dataset_label_report,
    )
    if sum(dataset_operations) > 1:
        raise SystemExit("Choose only one dataset operation per command.")
    if args.dataset_label is not None and args.dataset_preview is None:
        raise SystemExit("--dataset-label requires --dataset-preview.")
    if (
        (args.labeler_split != "all" or args.labeler_session is not None)
        and not args.dataset_labeler
    ):
        raise SystemExit("--labeler-split and --labeler-session require --dataset-labeler.")
    if args.dataset_init is not None:
        create_dataset_structure(args.dataset_init)
        print(f"Dataset structure ready: {args.dataset_init}")
        return

    if args.dataset_validate is not None:
        report = validate_dataset(args.dataset_validate)
        print(json.dumps(report.to_dict(), indent=2))
        if not report.valid:
            raise SystemExit(1)
        return

    if args.dataset_preview is not None:
        dataset_dir = None
        if args.dataset_label is None:
            image_path = args.dataset_preview.resolve()
            if image_path.parent.parent.name == "images":
                dataset_dir = image_path.parent.parent.parent
        preview = render_dataset_preview(
            args.dataset_preview,
            label_path=args.dataset_label,
            dataset_dir=dataset_dir,
        )
        cv2.imshow("Dataset label preview (read-only)", preview)
        try:
            while cv2.waitKey(0) & 0xFF not in (ord("q"), 27):
                pass
        finally:
            cv2.destroyAllWindows()
        return

    if args.dataset_label_report:
        print(json.dumps(build_labeling_report(args.dataset_dir), indent=2))
        return

    if args.dataset_labeler:
        run_dataset_labeler(
            args.dataset_dir,
            split=args.labeler_split,
            session_id=args.labeler_session,
        )
        return

    if args.dataset_capture:
        dataset_config = DatasetCaptureConfig(
            capture=CaptureConfig(
                monitor_index=args.monitor,
                max_fps=args.fps,
                region=args.region,
            ),
            sample_interval=args.sample_interval,
            max_samples=args.max_samples,
            output_dir=args.dataset_dir,
            session_id=args.dataset_session,
            split=None if args.dataset_split == "auto" else args.dataset_split,
            deduplicate=args.deduplicate,
            similarity_threshold=args.similarity_threshold,
        )
        report = record_dataset_session(dataset_config)
        print(json.dumps({
            "session_id": report.session_id,
            "split": report.split,
            "captured_frames": report.captured_frames,
            "sampled_frames": report.sampled_frames,
            "saved_frames": report.saved_frames,
            "duplicate_frames_skipped": report.duplicate_frames_skipped,
            "manifest_path": report.manifest_path.as_posix(),
            "session_metadata_path": report.session_metadata_path.as_posix(),
        }, indent=2))
        return

    if args.video_dataset_capture is not None:
        video_config = VideoDatasetCaptureConfig(
            video_path=args.video_dataset_capture,
            sample_interval=args.video_sample_interval,
            max_frames=args.video_max_frames,
            output_dir=args.dataset_dir,
            session_id=args.dataset_session,
            split=None if args.dataset_split == "auto" else args.dataset_split,
            deduplicate=args.deduplicate,
            similarity_threshold=args.similarity_threshold,
        )
        report = extract_video_dataset_session(video_config)
        print(json.dumps({
            "session_id": report.session_id,
            "split": report.split,
            "source_video": report.video_metadata.source_video,
            "duration_seconds": report.video_metadata.duration_seconds,
            "source_fps": report.video_metadata.source_fps,
            "source_resolution": {
                "width": report.video_metadata.width,
                "height": report.video_metadata.height,
            },
            "total_source_frames": report.video_metadata.total_frames,
            "source_frames_read": report.source_frames_read,
            "sampled_frames": report.sampled_frames,
            "saved_frames": report.saved_frames,
            "duplicate_frames_skipped": report.duplicate_frames_skipped,
            "labels_generated": False,
            "manifest_path": report.manifest_path.as_posix(),
            "session_metadata_path": report.session_metadata_path.as_posix(),
        }, indent=2))
        return

    vision_options = (
        args.vision_debug
        or args.vision_ocr
        or args.save_vision_debug
        or args.vision_image is not None
    )
    if vision_options:
        if args.vision_image is None:
            raise SystemExit(
                "--vision-debug and --save-vision-debug require --vision-image."
            )
        if args.save_frame is not None or args.save_first_frame or args.frames is not None:
            raise SystemExit(
                "Capture options cannot be combined with --vision-image."
            )
        frame = cv2.imread(str(args.vision_image), cv2.IMREAD_COLOR)
        if frame is None:
            raise OSError(f"Could not read vision input image: {args.vision_image}")
        context = load_frame_context(args.vision_image)
        detector = (
            VisionDetector(prompt_detector=PromptDetector(pytesseract_reader))
            if args.vision_ocr
            else None
        )
        game_state = estimate_game_state(frame, context, detector)
        print(json.dumps(game_state.to_dict(), indent=2))

        if args.save_vision_debug:
            image_path, state_path = save_vision_debug(
                frame, game_state, args.vision_output_dir
            )
            get_logger(__name__).info(
                "Saved vision debug frame: %s (GameState: %s).",
                image_path,
                state_path,
            )
        if args.vision_debug:
            cv2.imshow("Dead As Disco AI - Vision Debug (read-only)", draw_debug_frame(frame, game_state))
            try:
                while cv2.waitKey(0) & 0xFF not in (ord("q"), 27):
                    pass
            finally:
                cv2.destroyAllWindows()
        return

    config = CaptureConfig(
        monitor_index=args.monitor,
        max_fps=args.fps,
        region=args.region,
    )
    config.validate()
    if args.save_frame is not None and args.save_first_frame:
        raise SystemExit("--save-frame and --save-first-frame cannot be combined.")
    if args.frames is not None and args.frames < 1:
        raise SystemExit("--frames must be at least 1.")
    logger = get_logger(__name__)
    logger.info(
        "Starting capture on monitor %d (maximum %.1f FPS).",
        config.monitor_index,
        config.max_fps,
    )
    run_live_capture(
        config,
        save_frame=args.save_frame,
        output_dir=args.output_dir,
        save_first_frame=args.save_first_frame,
        max_frames=args.frames,
        show_preview=not args.no_preview,
    )


if __name__ == "__main__":
    main()
