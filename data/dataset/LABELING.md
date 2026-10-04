# Dataset labeling guide

## Classes

Class IDs are fixed by `dataset.yaml`:

| ID | Name | Label only when |
| ---: | --- | --- |
| 0 | `player` | The player character is visible and distinguishable from enemies. |
| 1 | `enemy` | An enemy is visible, without enough evidence to assign a special state. |
| 2 | `stunned_enemy` | A visible enemy is clearly stunned (for example, an unambiguous game animation or prompt); otherwise use `enemy`. |
| 3 | `airborne_enemy` | A visible enemy is clearly airborne. |
| 4 | `action_prompt` | A contextual action prompt is visible and can be boxed as a visual object. Do not infer its action meaning from E/F alone. |
| 5 | `takedown_indicator` | The takedown availability indicator is clearly visible. |

Do not create labels from the current yellow-color heuristic. No automatic
labels are generated. If a frame is ambiguous, leave its label file absent and
record/review it rather than guessing.

## Gameplay collection checklist

Collect naturally occurring, varied examples across separate bounded
sessions. Do not use game input to create a situation. Tick an item only after
reviewing collected frames:

- [ ] **Player:** standing, walking, sprinting, attacking, and dodging.
- [ ] **Player variation:** different screen positions and backgrounds; partial
  occlusion.
- [ ] **Enemy:** single and multiple enemies; near and far; visibly different
  enemy types where distinguishable.
- [ ] **Enemy variation:** attacking, moving, and partially occluded enemies.
- [ ] **Stunned enemy:** clearly confirmed stunned state at different distances
  and enemy types where applicable.
- [ ] **Airborne enemy:** clearly airborne examples at different heights and
  distances.
- [ ] **Action prompt:** E, F, and other contextual prompts only if observed;
  varied prompt positions.
- [ ] **Takedown indicator:** clearly visible and not visible in different
  combat situations.

Prefer multiple sessions (ordinary gameplay, combat-heavy play, multiple
enemies, naturally observed stunned/airborne cases, and prompts/takedowns).
Record the session ID and split for each session. Assign the entire continuous
session to one split; never distribute adjacent frames from one sequence
across train, validation, and test. The 70/20/10 target is approximate and
must not override session boundaries.

## YOLO detection label format

Create one UTF-8 `.txt` beside each labeled image using the same basename:

```text
class_id x_center y_center width height
```

Use one object per line. Coordinates are normalized to the image width/height
and must lie in `[0, 1]`; boxes must have positive width/height and remain
inside the image. Example only (not a game annotation):

```text
0 0.500000 0.500000 0.100000 0.300000
```

Empty label files are valid for reviewed images with no target-class objects.
An absent label file means "not labeled/reviewed yet".

## Directory and split rules

Keep matching image and label splits:

```text
images/{train,val,test}/<session>_<frame>_<timestamp>.png
labels/{train,val,test}/<same-basename>.txt
```

Assign all frames from one continuous gameplay session to a single split;
never randomly scatter frames from the same session among train, validation,
and test. Aim for approximately 70% train, 20% validation, and 10% test by
sessions (or grouped gameplay sequences), not by individual near-duplicate
frames. Automatic session assignment is deterministic but approximate; review
and adjust whole sessions if the collection is small.

## Labeling quality review

- [ ] Every saved frame has been manually reviewed; missing labels are not
  mistaken for negative examples.
- [ ] Boxes tightly surround only visible objects; no hidden or guessed objects
  are labeled.
- [ ] UI is not labeled as a player or enemy.
- [ ] An enemy is labeled `stunned_enemy` only when the state is visually
  confirmed, and `airborne_enemy` only when clearly airborne.
- [ ] Action-prompt boxes cover the visible prompt, without assigning an
  unverified meaning to E/F.
- [ ] Takedown boxes cover the actual visible indicator.
- [ ] Difficult cases and visual variation are represented; no class depends
  on one nearly identical scene.
- [ ] Previewed labels align with the source frame and pass dataset validation.
- [ ] Complete sessions remain in only one split and split leakage is absent.
- [ ] Class counts, images with multiple classes, reviewed empty images, and
  exact/near duplicates have been reviewed before a readiness decision.
