# Phase 4D — Independent validation/test gameplay capture

## Purpose and isolation

Object-detector validation and test images must come from complete gameplay
sessions not used to curate/train the detector. Splitting neighboring frames
from the same play sequence leaks near-identical backgrounds, poses, and
effects across the partitions, making evaluation over-optimistic. Capture each
recording as one indivisible session. Do not reassign individual frames later.

Capture destination is fixed at `data/dataset_independent_eval/`, separate
from both the six-class `data/dataset/` legacy source and
`data/dataset_player_enemy_v2/`. Its two-class YAML is for player/enemy
evaluation only. Validation purpose always maps to `images/val`; test purpose
always maps to `images/test`. No training choice is exposed by this capture
command. Runtime path guards reject roots overlapping either protected
dataset. Each session ID is unique, and an existing ID cannot be overwritten.

## How to record each session

1. Start the game yourself, bring the game to the foreground, and ensure the
   captured game area is unobscured. The recorder only reads screen pixels; it
   does not activate or control the game.
2. Run exactly one of the purpose-specific commands below. Keep the full
   continuous recording in that purpose. Do not combine validation/test takes
   from the same continuous gameplay sequence.
3. The preview starts immediately. Press **Q** or **Esc** in the preview to
   stop early, or **Ctrl+C** in the terminal. Capture also stops automatically
   at the 15-minute maximum. No game input is sent.
4. Add natural gameplay variation by playing normally; do not force a rare
   combat situation. A good recording includes ordinary movement, changes in
   camera distance/position, and whatever single/multiple-enemy encounters
   occur naturally.

### Recording targets and approximate duration

Target **10–15 minutes per session** at a maximum capture rate of **30 FPS**
and a sampling interval of **0.5 seconds** (up to about 1,200–1,800 sampled
frames if captured for the full duration). This is a collection target, not a
promise of object counts. Approximate review targets are at least 50 player
instances and 75 enemy instances in validation, and the same approximate
counts in test. Instances must be confirmed later through manual review; a
recorded frame count is not an object count.

The test session should be recorded separately and preferably later than
validation. Keep both sessions independent from each other and from the two
existing training sessions. The test set must stay unused for tuning until
final evaluation.

## Commands to record

Use the project virtual environment from the repository root. Replace the
session IDs with fresh, unique values for each recording.

### A. Validation session

```powershell
.\.venv\Scripts\python.exe -m app.main --independent-capture validation --independent-session-id validation_20261005_01 --monitor 1 --region 0,0,1536,864 --fps 30 --sample-interval 0.5 --capture-duration 900 --session-notes "Independent validation gameplay"
```

### B. Test session

```powershell
.\.venv\Scripts\python.exe -m app.main --independent-capture test --independent-session-id test_20261005_01 --monitor 1 --region 0,0,1536,864 --fps 30 --sample-interval 0.5 --capture-duration 900 --session-notes "Independent held-out test gameplay"
```

If the game is not at 1536x864 or is on another monitor, measure the correct
monitor-relative capture rectangle and update `--monitor`/`--region`. Do not
capture desktop areas that obscure or clip the game.

## Output and metadata

Each frame is stored under the purpose's fixed YOLO split:

```text
data/dataset_independent_eval/
  images/train/                         # stays empty in Phase 4D
  images/val/<session>_frame_*.png      # validation purpose only
  images/test/<session>_frame_*.png     # test purpose only
  labels/{train,val,test}/              # initially empty
  metadata/manifest.jsonl               # one source/session row per image
  metadata/<purpose>_sessions.jsonl     # one session summary per capture
  metadata/<purpose>/<session>/session.json
  images/{val,test}/<frame>.json        # frame metadata sidecar
```

Each frame record/sidecar includes timestamp, frame number, actual resolution,
capture FPS measurement, capture bounds, `source=live_screen_capture`,
`source_video=null` (live capture has no source video), session ID, purpose,
and fixed `val`/`test` split. The session manifest
records session ID, purpose, source, resolution, configured/measured FPS,
sample interval, saved frame count, start/end timestamps, notes, and whether
labels were generated. Capture never creates labels. Generated recordings and
metadata are local data and are ignored by Git; the versioned schema and
manifest example remain tracked.

## Later manual review (not part of capture)

After both sessions are complete, manually inspect frames with the two-class
future schema only. Do not copy the legacy six-class labels into this
destination or turn `stunned_enemy`, `airborne_enemy`, prompts, or icons into
`enemy` labels by assumption. Use the existing manual labeler against the
independent dataset root, selecting only the relevant split:

```powershell
.\.venv\Scripts\python.exe -m app.main --dataset-labeler --dataset-dir data\dataset_independent_eval --labeler-split val
.\.venv\Scripts\python.exe -m app.main --dataset-labeler --dataset-dir data\dataset_independent_eval --labeler-split test
```

This later labeling step is documented for future work only; do not start it
as part of Phase 4D. Keep test annotations and frames held out from threshold
tuning. Run dataset validation after manual review, and keep all frames from
each session in their assigned split.
