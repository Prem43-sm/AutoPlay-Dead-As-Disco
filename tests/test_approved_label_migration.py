import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from data.approved_label_migration import (
    apply_migration_plan,
    prepare_migration_plan,
)


class ApprovedLabelMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "legacy"
        self.destination = self.root / "future"
        (self.source / "images" / "train").mkdir(parents=True)
        (self.source / "labels" / "train").mkdir(parents=True)
        (self.source / "metadata").mkdir(parents=True)
        (self.destination / "images" / "train").mkdir(parents=True)
        (self.destination / "labels" / "train").mkdir(parents=True)
        (self.destination / "dataset.yaml").write_text(
            "names:\n  0: player\n  1: enemy\n", encoding="utf-8"
        )
        (self.source / "dataset.yaml").write_text(
            "names:\n  0: player\n  1: enemy\n  2: stunned_enemy\n"
            "  3: airborne_enemy\n  4: action_prompt\n  5: takedown_indicator\n",
            encoding="utf-8",
        )
        self.image_rel = "images/train/session_frame_000001_time.png"
        self.label_rel = "labels/train/session_frame_000001_time.txt"
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)
        self.image_bytes = encoded.tobytes()
        self.label_bytes = (
            "0 0.5 0.5 0.2 0.4\n"
            "1 0.7 0.5 0.2 0.4\n"
            "2 0.3 0.5 0.2 0.4\n"
        ).encode()
        (self.source / self.image_rel).write_bytes(self.image_bytes)
        (self.source / self.label_rel).write_bytes(self.label_bytes)
        (self.source / "metadata" / "manifest.jsonl").write_text(
            json.dumps(
                {
                    "image_path": self.image_rel,
                    "source_session": "session",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.approvals = self.root / "approvals.json"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_approvals(self, boxes: list[dict], split: str = "train") -> None:
        self.approvals.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "approvals": [
                        {
                            "source_image": self.image_rel,
                            "source_label": self.label_rel,
                            "image_sha256": hashlib.sha256(self.image_bytes).hexdigest(),
                            "label_sha256": hashlib.sha256(self.label_bytes).hexdigest(),
                            "target_split": split,
                            "approved_by": "reviewer",
                            "reviewed_at": "2026-10-05T00:00:00Z",
                            "approved_boxes": boxes,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_plan_includes_only_explicit_player_and_enemy_approvals(self) -> None:
        self.write_approvals(
            [
                {"line_number": 1, "class_name": "player"},
                {"line_number": 2, "class_name": "enemy"},
            ]
        )

        plan = prepare_migration_plan(self.source, self.destination, self.approvals)

        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].box_count, 2)
        self.assertEqual(
            plan[0].label_bytes,
            b"0 0.5 0.5 0.2 0.4\n1 0.7 0.5 0.2 0.4\n",
        )
        self.assertFalse((self.destination / self.image_rel).exists())
        self.assertEqual((self.source / self.label_rel).read_bytes(), self.label_bytes)

    def test_state_class_cannot_be_converted_to_enemy(self) -> None:
        self.write_approvals([{"line_number": 3, "class_name": "enemy"}])

        with self.assertRaisesRegex(ValueError, "does not match the legacy class"):
            prepare_migration_plan(self.source, self.destination, self.approvals)

    def test_invalid_source_box_is_refused(self) -> None:
        self.label_bytes = b"0 0.5 0.5 1.2 0.4\n"
        (self.source / self.label_rel).write_bytes(self.label_bytes)
        self.write_approvals([{"line_number": 1, "class_name": "player"}])

        with self.assertRaisesRegex(ValueError, "invalid legacy box"):
            prepare_migration_plan(self.source, self.destination, self.approvals)

    def test_source_checksum_change_is_refused(self) -> None:
        self.write_approvals([{"line_number": 1, "class_name": "player"}])
        (self.source / self.label_rel).write_text("0 0.4 0.5 0.2 0.4\n")

        with self.assertRaisesRegex(ValueError, "label checksum changed"):
            prepare_migration_plan(self.source, self.destination, self.approvals)

    def test_apply_copies_only_approved_box_and_never_overwrites(self) -> None:
        self.write_approvals([{"line_number": 1, "class_name": "player"}])
        plan = prepare_migration_plan(self.source, self.destination, self.approvals)

        apply_migration_plan(self.destination, plan)

        self.assertEqual(
            (self.destination / "labels/train/session_frame_000001_time.txt").read_bytes(),
            b"0 0.5 0.5 0.2 0.4\n",
        )
        self.assertEqual(
            (self.destination / "images/train/session_frame_000001_time.png").read_bytes(),
            self.image_bytes,
        )
        with self.assertRaises(FileExistsError):
            prepare_migration_plan(self.source, self.destination, self.approvals)

    def test_session_cannot_be_assigned_to_multiple_splits(self) -> None:
        image_rel_2 = "images/train/session_frame_000002_time.png"
        label_rel_2 = "labels/train/session_frame_000002_time.txt"
        (self.source / image_rel_2).write_bytes(self.image_bytes)
        (self.source / label_rel_2).write_bytes(self.label_bytes)
        with (self.source / "metadata" / "manifest.jsonl").open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"image_path": image_rel_2, "source_session": "session"}
                )
                + "\n"
            )
        first = {
            "source_image": self.image_rel,
            "source_label": self.label_rel,
            "image_sha256": hashlib.sha256(self.image_bytes).hexdigest(),
            "label_sha256": hashlib.sha256(self.label_bytes).hexdigest(),
            "target_split": "train",
            "approved_by": "reviewer",
            "reviewed_at": "2026-10-05T00:00:00Z",
            "approved_boxes": [{"line_number": 1, "class_name": "player"}],
        }
        second = dict(first)
        second.update(
            {
                "source_image": image_rel_2,
                "source_label": label_rel_2,
                "target_split": "val",
            }
        )
        self.approvals.write_text(
            json.dumps({"schema_version": 1, "approvals": [first, second]}),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "multiple destination splits"):
            prepare_migration_plan(self.source, self.destination, self.approvals)


if __name__ == "__main__":
    unittest.main()
