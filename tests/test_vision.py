import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from game.game_state import GameStatus
from vision.enemy_detector import EnemyDetector
from vision.hud_detector import HudDetector
from vision.models import BoundingBox, Detection, FrameContext
from vision.player_detector import PlayerDetector
from vision.prompt_detector import PromptDetector
from vision.state_estimator import (
    draw_debug_frame,
    estimate_game_state,
    save_vision_debug,
)


class DetectionModelTests(unittest.TestCase):
    def test_bbox_center_and_clip(self) -> None:
        box = BoundingBox(-4, 5, 20, 10)
        self.assertEqual(box.center, (6.0, 10.0))
        self.assertEqual(box.clip(12, 12), BoundingBox(0, 5, 12, 7))
        self.assertIsNone(BoundingBox(20, 20, 5, 5).clip(12, 12))

    def test_bbox_rejects_empty_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            BoundingBox(0, 0, 0, 10)

    def test_detection_validates_confidence_and_has_frame_context(self) -> None:
        detection = Detection(
            "enemy",
            BoundingBox(1, 2, 3, 4),
            0.8,
            "test",
            frame_number=2,
        )
        self.assertEqual(detection.frame_number, 2)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            Detection("enemy", BoundingBox(1, 2, 3, 4), 1.1, "test")


class PlayerDetectorTests(unittest.TestCase):
    def test_detects_synthetic_yellow_clothing_candidate(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (120, 70), (180, 170), (0, 255, 255), -1)
        context = FrameContext(
            datetime(2026, 10, 4, tzinfo=timezone.utc), frame_number=3
        )

        result = PlayerDetector().detect(frame, context)

        self.assertEqual(result.status, "detected")
        self.assertEqual(len(result.detections), 1)
        detection = result.detections[0]
        self.assertEqual(detection.class_name, "player")
        self.assertEqual(detection.frame_number, 3)
        self.assertIn("garment candidate", detection.limitation or "")
        self.assertGreater(detection.confidence, 0)

    def test_no_matching_pixels_is_not_detected(self) -> None:
        result = PlayerDetector().detect(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual(result.status, "not_reliably_detected")
        self.assertEqual(result.detections, ())


class EnemyHudPromptTests(unittest.TestCase):
    def test_enemy_detector_does_not_claim_unverified_detection(self) -> None:
        result = EnemyDetector().detect(np.zeros((80, 120, 3), dtype=np.uint8))
        self.assertEqual(result.status, "not_reliably_detected")
        self.assertEqual(result.detections, ())
        self.assertIn("NOT RELIABLY DETECTED YET", result.limitation or "")

    def test_hud_detector_finds_synthetic_upper_left_cue_only(self) -> None:
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        frame[10:40, 10:70] = (255, 0, 255)

        result = HudDetector().detect(frame)

        self.assertEqual(result.status, "detected")
        self.assertEqual(result.detections[0].class_name, "hud_region_candidate")
        self.assertIn("not decoded", result.limitation or "")

    def test_hud_detector_returns_unknown_when_cue_is_absent(self) -> None:
        result = HudDetector().detect(np.zeros((200, 300, 3), dtype=np.uint8))
        self.assertEqual(result.status, "not_reliably_detected")

    def test_prompt_detector_is_unknown_without_optional_ocr(self) -> None:
        result = PromptDetector().detect(np.zeros((50, 80, 3), dtype=np.uint8))
        self.assertEqual(result.status, "not_reliably_detected")
        self.assertEqual(result.detections, ())

    def test_prompt_detector_extracts_key_but_not_action_meaning(self) -> None:
        frame = np.zeros((50, 80, 3), dtype=np.uint8)
        reader = lambda _: [("Press F", BoundingBox(50, 20, 20, 20), 0.93)]

        result = PromptDetector(reader).detect(frame)

        self.assertEqual(result.status, "detected")
        self.assertEqual(result.detections[0].class_name, "prompt:F")
        self.assertIn("contextual", result.detections[0].limitation or "")

    def test_prompt_detector_ignores_low_confidence_ocr(self) -> None:
        frame = np.zeros((50, 80, 3), dtype=np.uint8)
        reader = lambda _: [("F", BoundingBox(50, 20, 10, 10), 0.2)]

        result = PromptDetector(reader).detect(frame)

        self.assertEqual(result.status, "not_detected")
        self.assertEqual(result.detections, ())


class GameStateAndDebugTests(unittest.TestCase):
    def test_game_state_preserves_unknown_fields_and_status(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        context = FrameContext(
            datetime(2026, 10, 4, tzinfo=timezone.utc), frame_number=8, capture_fps=26.5
        )

        state = estimate_game_state(frame, context)

        self.assertEqual(state.timestamp, context.timestamp.isoformat())
        self.assertEqual(state.frame_number, 8)
        self.assertEqual(state.capture_fps, 26.5)
        self.assertFalse(state.player.detected)
        self.assertIsNone(state.enemies.count)
        self.assertIsNone(state.hud.health)
        self.assertIsNone(state.hud.fever)
        self.assertIsNone(state.hud.takedown_available)
        self.assertIsNone(state.hud.score)
        self.assertIsNone(state.prompt.visible)
        self.assertEqual(state.game_status, GameStatus.UNKNOWN)
        json.dumps(state.to_dict())

    def test_debug_frame_and_bounded_serialization(self) -> None:
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50, 20), (100, 80), (0, 255, 255), -1)
        state = estimate_game_state(frame)

        debug = draw_debug_frame(frame, state)
        self.assertEqual(debug.shape, frame.shape)

        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path, state_path = save_vision_debug(
                frame, state, Path(temporary_directory)
            )
            self.assertTrue(image_path.is_file())
            self.assertTrue(state_path.is_file())
            saved_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_state["game_status"], GameStatus.UNKNOWN.value)
            self.assertEqual(len(list(Path(temporary_directory).glob("*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
