# Test Set Freeze Record

## Dataset
- Dataset: Common Voice Beijing26 zh-CN
- Source candidates: test_candidates.csv
- Original manifest: manifest.csv
- Dev annotation source: dev_candidates_W7D5.csv
- Original Test Manifest samples: 170
- Frozen Test samples: 98

## Sampling
- Method: simple_random
- Seed: 42
- Selection rank: 1-98

## Manual Verification
- Verified: 98
- Pending: 0
- Missing review_minutes: 10
- Missing review_minutes were not imputed.

## Duration
- Minimum: 5.021625 seconds
- Maximum: 13.147500 seconds
- Required range: 5-30 seconds

## Uniqueness
- candidate_uid: unique
- audio_name: unique
- parent_uid: unique
- selection_rank: unique

## Provenance
- Outside original Test Manifest: 0
- Missing audio files: 0

## Dev-Test Overlap
- candidate_uid: 0
- audio_name: 0
- parent_uid: 0
- speaker_id: 0

## Evaluation
- Reference field: score_text
- raw_text retained for annotation auditing.

## Freeze
- Freeze time: 2026-11-26T01:43:11-06:00
- SHA-256: f6b12ed6c0f2ee9bd46a06b00620eadee72d0cb29b2d1225ddc9df3d69cdf2a1
- File: test_manifest.csv

## Policy
- Frozen size: 98, not 100.
- No sample will be added or removed based on model results.
- Missing review_minutes values are retained as missing.
