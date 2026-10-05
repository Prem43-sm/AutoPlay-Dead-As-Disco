# Independent validation/test capture dataset

This isolated dataset is reserved for whole-session validation and test
captures. Its class schema is `player`, `enemy`; it contains no captured
images or labels initially. Capture assigns `validation` purpose only to
`images/val` and `test` purpose only to `images/test`. Training data must not
be placed here or sourced from these sessions.

See [`../../docs/PHASE_4D_CAPTURE.md`](../../docs/PHASE_4D_CAPTURE.md) for
recording instructions, safeguards, session metadata, and later manual review.
