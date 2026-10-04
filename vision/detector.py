import numpy as np
from numpy.typing import NDArray

from game.game_state import (
    EnemiesState,
    GameState,
    HudState,
    PlayerState,
    PromptState,
)
from vision.enemy_detector import EnemyDetector
from vision.hud_detector import HudDetector
from vision.models import Detection, FrameContext
from vision.player_detector import PlayerDetector
from vision.prompt_detector import PromptDetector


class VisionDetector:
    def __init__(
        self,
        player_detector: PlayerDetector | None = None,
        enemy_detector: EnemyDetector | None = None,
        hud_detector: HudDetector | None = None,
        prompt_detector: PromptDetector | None = None,
    ) -> None:
        self.player_detector = player_detector or PlayerDetector()
        self.enemy_detector = enemy_detector or EnemyDetector()
        self.hud_detector = hud_detector or HudDetector()
        self.prompt_detector = prompt_detector or PromptDetector()

    def process(
        self, frame: NDArray[np.uint8], context: FrameContext | None = None
    ) -> GameState:
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("Vision pipeline expects a color image.")
        context = context or FrameContext.now(frame_number=1)
        player_result = self.player_detector.detect(frame, context)
        enemy_result = self.enemy_detector.detect(frame, context)
        hud_result = self.hud_detector.detect(frame, context)
        prompt_result = self.prompt_detector.detect(frame, context)

        player_detection = (
            player_result.detections[0] if player_result.detections else None
        )
        hud_detection = hud_result.detections[0] if hud_result.detections else None
        prompt_detection = (
            prompt_result.detections[0] if prompt_result.detections else None
        )
        all_detections: tuple[Detection, ...] = (
            player_result.detections
            + enemy_result.detections
            + hud_result.detections
            + prompt_result.detections
        )
        notes = tuple(
            note
            for note in (
                player_result.limitation,
                enemy_result.limitation,
                hud_result.limitation,
                prompt_result.limitation,
            )
            if note
        )
        return GameState(
            timestamp=context.timestamp.isoformat(),
            frame_number=context.frame_number,
            capture_fps=context.capture_fps,
            player=PlayerState(
                detected=player_detection is not None,
                bbox=player_detection.bbox if player_detection else None,
                center=player_detection.bbox.center if player_detection else None,
                confidence=player_detection.confidence if player_detection else None,
                limitation=player_result.limitation,
            ),
            enemies=EnemiesState(
                count=(
                    len(enemy_result.detections)
                    if enemy_result.status != "not_reliably_detected"
                    else None
                ),
                detections=enemy_result.detections,
                centers=tuple(d.bbox.center for d in enemy_result.detections),
                confidence=enemy_result.confidence,
                limitation=enemy_result.limitation,
            ),
            hud=HudState(
                visible=hud_detection is not None,
                confidence=hud_detection.confidence if hud_detection else None,
                region=hud_detection.bbox if hud_detection else None,
                limitation=hud_result.limitation,
            ),
            prompt=PromptState(
                visible=(
                    True if prompt_detection else
                    False if prompt_result.status == "not_detected" else
                    None
                ),
                key=(
                    prompt_detection.class_name.partition(":")[2]
                    if prompt_detection
                    else None
                ),
                bbox=prompt_detection.bbox if prompt_detection else None,
                confidence=prompt_detection.confidence if prompt_detection else None,
                limitation=prompt_result.limitation,
            ),
            detections=all_detections,
            notes=notes,
        )
