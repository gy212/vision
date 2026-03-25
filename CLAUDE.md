# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MediaPipe-based real-time pose and hand gesture recognition system with template-based action matching. Uses DTW (Dynamic Time Warping) for pose sequence comparison. Chinese-localized desktop UI via Tkinter.

## Common Commands

### Installation
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Run Desktop UI
```powershell
.\.venv\Scripts\python.exe apps/app_ui.py
```

### Run CLI (camera)
```powershell
.\.venv\Scripts\python.exe apps/main.py --source 0
```

### Offline video with multi-threading
```powershell
.\.venv\Scripts\python.exe apps/main.py --source input.mp4 --pose heavy --out out.mp4 --no-show --workers 4
```

### Create pose template from video
```powershell
.\.venv\Scripts\python.exe apps/make_template.py --video action.mp4 --pose heavy --preview
```

### Match template against video
```powershell
.\.venv\Scripts\python.exe apps/match_template.py --template template.npz --video test.mp4 --preview
```

### Batch dual-template comparison
```powershell
.\.venv\Scripts\python.exe batch/batch_dual_compare.py --standard_dir "标准样本" --student_dir "学员样本" --pose full --out_dir "输出目录" --rules --action both
```

### Syntax check (no formal test suite)
```powershell
.\.venv\Scripts\python.exe -m py_compile .\apps\main.py .\apps\app_ui.py .\core\vision_pipeline.py .\core\pose_features.py .\core\action_compare.py .\apps\make_template.py .\apps\match_template.py .\core\video_writer.py .\batch\batch_dual_compare.py .\core\rule_scoring.py
```

## Architecture

### Module Responsibilities

- **core/vision_pipeline.py**: Core MediaPipe wrapper (`MediaPipePipeline` class) handling pose/hand landmarking, frame annotation, and action classification (HANDS_UP, SQUAT, V-sign detection)
- **core/pose_features.py**: Pose normalization (translation/scale/rotation invariant) and DTW subsequence matching algorithm
- **core/action_compare.py**: High-level template matching - creates templates from videos and compares against targets; includes dual-template (front+side) comparison with automatic view splitting
- **core/rule_scoring.py**: Rule-based scoring engine using raw Pose33 landmarks; computes violation ratios and deductions per scoring guide
- **core/video_writer.py**: Codec-agnostic video output with fallback chain (H.264 → MJPEG → XVID)
- **batch/batch_dual_compare.py**: Directory-level batch processing for dual-template comparison; outputs CSV/JSONL results
- **apps/app_ui.py**: Tkinter GUI with two windows (main preview + compare dialog), handles threading for non-blocking video processing
- **apps/main.py**: CLI entry point supporting camera/video sources with optional export
- **apps/make_template.py / match_template.py**: CLI utilities for template creation and matching

### Processing Modes

- **VIDEO mode**: Temporal tracking with smoothing for live camera/streams. Single-threaded, stateful.
- **IMAGE mode**: Stateless per-frame processing for offline batch operations. Multi-threaded for throughput.

### Threading Model (offline processing)

1. Reader thread fills input frame queue
2. Worker threads process frames independently (IMAGE mode)
3. Main thread reorders results and writes output

### Pose Normalization Pipeline (pose_features.py)

Landmarks 11-32 (excludes face) → translate by hip center → scale by torso length (v3) or shoulder width (v1/v2) → rotate to align shoulders/torso → output (22, 2) tensor

Multiple normalizer versions exist (`normalize_pose_xy_v1`, `_v3`); templates store `feature_layout` to auto-select the matching normalizer at comparison time.

### Dual-Template Comparison Pipeline

For comparing a student video (containing front+side views) against standard front/side templates:

1. Extract pose features from student video with per-frame "frontness" score
2. Auto-split into front/side segments by median threshold on frontness
3. Extract representative repetition from each standard template via motion-energy autocorrelation
4. Multi-match subsequence DTW on each segment; aggregate scores with trimmed mean (drop max/min if ≥3 reps)
5. Combine front/side scores by weighted average → output integer percent (0-100)
6. Optional rule-based deductions via `rule_scoring.py`

## Coding Conventions

- Python 4-space indentation, typed functions where practical
- UI code in `app_ui.py`, inference logic in `vision_pipeline.py`
- Models auto-download to `models/` directory on first run (gitignored)
- Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`

## 任务完成规范

- **每次完成任务后，必须将修改总结写入 `change.md` 文件**
- 记录内容应包括：修改日期、问题描述、修改内容、验证方法
