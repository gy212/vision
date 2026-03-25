# Dual-Template Action Compare (Front + Side)

Goal: compare a student's single MP4 (contains **front** + **side** view) against two standard videos (front/side),
and output a **0-100 integer** similarity score (higher = closer to standard).

This repo already has template-based pose matching via subsequence DTW. The implementation extends it with:

- Automatic front/side split for the student's long video
- Scoring multiple repetitions and aggregating with "drop max & min, then average"

## Pipeline

1) Pose feature extraction
- MediaPipe PoseLandmarker (VIDEO mode)
- Per-frame normalized features: 22 landmarks (indices 11..32), (x,y), centered/scaled/rotated

2) Student front/side split (no manual turn timestamp required)
- Compute a per-frame "frontness" score from raw pose landmarks:
  - shoulder width normalized by torso length
  - left/right visibility balance
- Smooth the score; split by median threshold
- Take the longest contiguous segment as **front**, and the longest complementary segment as **side**
- Shrink segment edges by a small margin to remove turning frames

3) Standard "one repetition" query (because the standard video repeats many times)
- Compute motion energy on the standard template features
- Estimate dominant period via normalized autocorrelation (energy signal)
- Pick a representative mid-cycle window as DTW query

4) Multi-repetition matching + trimmed mean
- For each view segment:
  - Restrict to the most active range (motion-energy heuristic)
  - Run greedy multi-match subsequence DTW:
    - find best match, exclude it, repeat
    - stop when matches become much worse than the best
  - Convert each match cost -> score with baseline normalization
  - Aggregate scores by trimmed mean: if >=3 reps, drop max & min then average

5) Final score
- Combine front/side scores by weighted average (default: front=0.4, side=0.6)
- Output `round(score * 100)` as integer percent

6) Optional rule-based scoring (from scoring guide)
- Uses raw Pose33 landmarks (no template matching) to evaluate rule deductions
- Computes per-view rule score and deduction totals
- Enabled via `compare_video_to_dual_templates(..., enable_rules=True, action_scope="both")`

## API

Implemented in `action_compare.py`:

- `compare_video_to_dual_templates(front_template_path, side_template_path, video_path, ...)`
  - Returns `DualCompareResult`
  - Use `DualCompareResult.combined_percent` as the final integer score
  - Rule outputs (when enabled): `front_rule_score`, `side_rule_score`, `front_rule_deduction`, `side_rule_deduction`,
    `front_rule_violations`, `side_rule_violations`

Templates are `.npz` files produced by the existing template creation workflow (UI or `make_template.py`).

## Tunables (when needed)

- `w_front`, `w_side`: front/side weight in final score
- `baseline`: cost->score mapping, `score = baseline / (baseline + avg_cost)`
- Cycle estimation window: min/max period range uses seconds (0.4s..3.0s) relative to FPS

## Notes / Limitations

- View split is heuristic; if the camera angle is not clearly front vs side, scores may be less stable.
- The method is training-free and robust to background/light changes, but depends on pose detection quality.
