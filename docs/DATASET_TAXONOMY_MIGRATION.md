# Phase 4C.2 — Taxonomy migration and label audit

## Dataset generations

### Legacy source (read-only for this migration)

`data/dataset/` remains the original dataset and its `dataset.yaml` remains the
six-class schema:

| Legacy ID | Class |
|---:|---|
| 0 | `player` |
| 1 | `enemy` |
| 2 | `stunned_enemy` |
| 3 | `airborne_enemy` |
| 4 | `action_prompt` |
| 5 | `takedown_indicator` |

Do not delete, rewrite, renumber, or automatically reinterpret its labels.
In particular, `stunned_enemy` and `airborne_enemy` are not silently promoted
to `enemy`; prompts and takedown icons are not object classes in the future
taxonomy.

### Future object detector (`data/dataset_player_enemy_v2/`)

The separate `dataset.yaml` is versioned by its containing directory and
defines only:

| Future ID | Class |
|---:|---|
| 0 | `player` |
| 1 | `enemy` |

The destination currently contains only this config, an empty approval
template, and workflow documentation; **no images or labels have been copied**.
Use HUD regions for hearts, enemy bars, meters, score/count/beat text; OCR and
templates for prompts and icons; and temporal tracking/state estimation for
attacking, stun/down, airborne, incoming-attack, counter/dodge windows, and
game/mission transitions. See
[`VISUAL_DETECTION_SPEC.md`](./VISUAL_DETECTION_SPEC.md) for detector details
and the proposed state schema.

## Invalid legacy boxes: manual audit

The source images are 1536x864. Original YOLO coordinates are reproduced
exactly from the legacy label line. Pixel bounds are derived from those
coordinates and are included to make manual review quicker. A box reported
outside the image may be an actual crop or a floating-point rounding sliver at
an edge; the original image must decide. **No coordinate correction is
authorized or applied by this report.**

For every row: open the listed image with its label overlay, verify the
class/object, then redraw the tight box around the visible subject within
`[0,1536] x [0,864]`. If the visible object reaches an image edge, set that
edge exactly to the boundary and recompute normalized coordinates from the
manually verified visible bounds; if it does not, draw to the actual silhouette
instead. Do not mechanically clip the old box.

| # | Image (relative to `data/dataset/`) | Class | Original YOLO line (`id xc yc w h`) | Derived pixel bounds (x1,y1)-(x2,y2) | Validator reason | Suggested manual correction |
|---:|---|---|---|---|---|---|
| 1 | `images/train/gameplay_20261004_1702_main_01_frame_000004_20261004T113250_554004Z.png` | player | `0 0.63541667 0.76504630 0.08854167 0.46990741` | (908,458)-(1044,864) | Extends beyond bottom boundary (may be edge-rounding) | Inspect feet/body at bottom; redraw visible player box with bottom at the image edge only if the visible silhouette is cropped there. |
| 2 | `images/train/gameplay_20261004_1702_main_01_frame_000005_20261004T113252_582755Z.png` | enemy | `1 0.98307292 0.72395833 0.03385417 0.43402778` | (1484,438)-(1536,813) | Extends beyond right boundary (may be edge-rounding) | Inspect the enemy at right edge; redraw its visible silhouette and set right edge exactly to 1536 only if cropped. |
| 3 | `images/train/gameplay_20261004_1702_main_01_frame_000007_20261004T113256_632312Z.png` | stunned_enemy | `2 0.27246094 0.79456019 0.22721354 0.41087963` | (244,509)-(593,864) | Extends beyond bottom boundary (may be edge-rounding) | Verify the subject and state; redraw the visible box at the bottom boundary only if the image actually crops it. |
| 4 | `images/train/gameplay_20261004_1702_main_01_frame_000009_20261004T113300_658449Z.png` | enemy | `1 0.84830729 0.61979167 0.28515625 0.76041667` | (1084,207)-(1522,864) | Extends beyond bottom boundary (may be edge-rounding) | Redraw the enemy silhouette; preserve x2=1522 only if it fits and use bottom=864 only if the subject is cropped. |
| 5 | `images/train/gameplay_20261004_1702_main_01_frame_000021_20261004T113324_961745Z.png` | player | `0 0.60677083 0.79803241 0.10937500 0.40393519` | (848,515)-(1016,864) | Extends beyond bottom boundary (may be edge-rounding) | Inspect lower legs/feet and redraw to the visible player extent, with exact bottom boundary only if cropped. |
| 6 | `images/train/gameplay_20261004_1702_main_01_frame_000023_20261004T113328_984413Z.png` | player | `0 0.45019531 0.77893519 0.25585938 0.44212963` | (495,482)-(888,864) | Extends beyond bottom boundary (may be edge-rounding) | Redraw player silhouette from source image; bottom=864 only if it reaches the frame edge. |
| 7 | `images/train/gameplay_20261004_1702_main_01_frame_000023_20261004T113328_984413Z.png` | enemy | `1 0.63834635 0.74942130 0.17513021 0.50115741` | (846,431)-(1115,864) | Extends beyond bottom boundary (may be edge-rounding) | Verify the enemy/player overlap and redraw only the enemy silhouette; bottom=864 only if actually cropped. |
| 8 | `images/train/gameplay_20261004_1702_main_01_frame_000024_20261004T113331_016714Z.png` | takedown_indicator | `5 0.39811198 0.86053241 0.25065104 0.27893519` | (419,623)-(804,864) | Extends beyond bottom boundary (may be edge-rounding) | This is UI, not a future YOLO class. Do not migrate it; if retaining legacy annotation, redraw the visible icon bounds within the frame. |
| 9 | `images/train/gameplay_20261004_1702_main_01_frame_000026_20261004T113335_079906Z.png` | player | `0 0.62565104 0.77893519 0.10937500 0.44212963` | (877,482)-(1045,864) | Extends beyond bottom boundary (may be edge-rounding) | Inspect the player crop and redraw tight visible bounds; bottom=864 only if the silhouette reaches it. |
| 10 | `images/train/gameplay_20261004_1702_main_01_frame_000026_20261004T113335_079906Z.png` | enemy | `1 0.11067708 0.76736111 0.22135417 0.46527778` | (0,462)-(340,864) | Extends beyond left boundary by rounding epsilon and bottom boundary (may be edge-rounding) | Left edge computes to approximately 0; redraw both edges from the image, using x1=0 only if cropped and bottom=864 only if cropped. |
| 11 | `images/train/gameplay_20261004_1702_main_01_frame_000028_20261004T113339_116322Z.png` | enemy | `1 0.86067708 0.80729167 0.22395833 0.38541667` | (1150,531)-(1494,864) | Extends beyond bottom boundary (may be edge-rounding) | Inspect the lower silhouette and redraw; bottom=864 only if the subject is cut by the image boundary. |

Some coordinate calculations display exactly at an edge because the validator
compares the original decimal floats, not rounded pixel coordinates. Preserve
the original file until a human has reviewed and explicitly corrected it.

## Explicit-approval migration workflow

`data/approved_label_migration.py` implements a fail-closed, non-destructive
copy workflow. It does not change the legacy dataset and does not infer
approvals. Dry-run is the default. The tool copies only individually approved,
valid source lines whose legacy class is exactly `player` or `enemy`; it never
maps state/UI classes to `enemy`.

1. Review an image and its existing label overlay manually. Correct invalid
   legacy annotations only through an independently reviewed legacy-label
   change; this migration utility never edits them.
2. Create a private copy of
   [`../data/dataset_player_enemy_v2/approvals.example.json`](../data/dataset_player_enemy_v2/approvals.example.json)
   and add one approval record per image. Specify source paths, the source
   image and label SHA-256 values, one-based line numbers and class names for
   each approved box, reviewer, UTC review time, and target split. Approve
   only visible, tight, valid player/enemy boxes. Do not add lines for legacy
   classes 2-5.
3. Keep every source session assigned to a single destination split. The
   utility reads source-session IDs from the legacy manifest and rejects any
   session assigned to more than one split.
4. Preview the exact proposed copies:

   ```powershell
   .\.venv\Scripts\python.exe -m data.approved_label_migration --source data\dataset --destination data\dataset_player_enemy_v2 --approvals path\to\reviewed-approvals.json
   ```

5. Inspect the dry-run image/label list and counts. Only after approval by the
   dataset owner, repeat with `--apply`. The tool verifies the checksums,
   source/session mapping, class/line match, box bounds, and destination
   nonexistence before writing. It refuses to overwrite existing files. Keep
   the approval manifest as provenance.
6. Run dataset validation against `data\dataset_player_enemy_v2`. Use the
   existing validator only after the standard dataset folders are present;
   do not change the legacy `data\dataset\dataset.yaml`.

Approval record shape:

```json
{
  "schema_version": 1,
  "approvals": [
    {
      "source_image": "images/train/example_frame.png",
      "source_label": "labels/train/example_frame.txt",
      "image_sha256": "<64 lowercase hex characters>",
      "label_sha256": "<64 lowercase hex characters>",
      "target_split": "train",
      "approved_by": "reviewer-id",
      "reviewed_at": "2026-10-05T00:00:00Z",
      "approved_boxes": [
        {"line_number": 1, "class_name": "player"}
      ]
    }
  ]
}
```

Hashes are mandatory so approvals cannot silently outlive source changes.
Each selected line must be valid against the source image dimensions. If an
image has both an eligible player/enemy box and an unapproved state/UI box,
only the explicitly selected line is copied. An approval for `stunned_enemy`,
`airborne_enemy`, `action_prompt`, or `takedown_indicator` is rejected.

## Complete-session split plan

Current state: both complete existing sessions (`gameplay_20261004_1702_main_01`
and `gameplay_video_20261004_01`) are in **train**. Keep them together; do not
split adjacent frames between partitions. Current `val` and `test` remain
empty. Session-level assignment for future additions:

| Partition | Existing sessions | Minimum next addition | Purpose |
|---|---|---|---|
| Train | Both current sessions | One new varied natural gameplay session if existing approved boxes do not supply the coverage targets below | Fit player/enemy detector; cover ordinary movement, varied scale, and single/multiple enemies |
| Validation | None | One complete new session, never sampled from either current recording | Tune thresholds and identify failure modes |
| Test | None | One later, complete, independent session kept untouched until evaluation | Final unbiased player/enemy evaluation |

Treat a source video or uninterrupted recording as an indivisible group. If
future recordings are closely related takes from the same play sequence, keep
them in one partition. Do not use random frame-level splits.

## Minimum additional data collection

These are practical minimum gates for a first detector evaluation, not
statistical guarantees. Count **manually approved visible object instances**,
not sampled frames. Both classes should appear across varied distances,
positions, poses, occlusion, and numbers of simultaneous enemies.

| Need | Minimum collection/review target |
|---|---|
| Player detection training | Use eligible, corrected existing player boxes only after review; target at least 200 approved player instances across at least 3 training sessions (including existing sessions if suitable). Collect another natural train session only if that count/diversity is not met. |
| Enemy detection training | Use eligible, corrected ordinary-enemy boxes only after review; target at least 300 approved enemy instances across the same or additional train sessions, including single and multiple enemies. Do not count stunned/airborne boxes until separately reviewed and approved as ordinary enemy instances. |
| Validation | Record at least one wholly separate session; target at least 50 player and 75 enemy instances, naturally occurring and manually approved. Do not tune on test data. |
| Test | Record at least one later, wholly separate session; target at least 50 player and 75 enemy instances. Freeze annotations and thresholds before using it for final evaluation. |

If a session does not naturally contain enough objects or variation, collect
more ordinary gameplay later. Do not force attacks, airborne states,
finishers, deaths, or other rare situations. These object-detector targets do
not validate the separate HUD/OCR/temporal detectors; use the event-specific
requirements in `VISUAL_DETECTION_SPEC.md` for those.
