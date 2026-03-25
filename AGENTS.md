# Repository Guidelines

## Project Structure & Module Organization

```
vision/
├── core/               # Core modules (MediaPipe pipeline, pose features, rules)
│   ├── vision_pipeline.py    # MediaPipe Tasks pipeline (PoseLandmarker + HandLandmarker)
│   ├── pose_features.py      # Pose normalization and DTW matching algorithms
│   ├── action_compare.py     # Template matching with dual-view comparison
│   ├── rule_scoring.py       # Rule-based scoring engine
│   └── video_writer.py       # Video output codec utilities
├── apps/               # Application entry points
│   ├── main.py               # CLI runner for camera/offline video processing
│   ├── app_ui.py             # Tkinter desktop UI (Chinese localized)
│   ├── make_template.py      # Create pose templates from videos
│   └── match_template.py     # Match templates against videos
├── batch/              # Batch processing tools
│   ├── batch_dual_compare.py # Dual-template comparison for directories
│   ├── batch_export_skeleton.py
│   └── batch_tech_eval.py
├── analysis/           # Analysis and evaluation tools
│   ├── tech_eval.py          # Technical evaluation core logic
│   ├── analyze_2mp4.py
│   └── ...
├── tests/              # Test scripts
├── models/             # Downloaded `.task` model files (gitignored)
└── requirements.txt    # Python dependencies
```

## Build, Test, and Development Commands

Create/install (from repo root):
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run UI:
```powershell
.\.venv\Scripts\python.exe apps/app_ui.py
```

Run CLI (camera `0`):
```powershell
.\.venv\Scripts\python.exe apps/main.py --source 0
```

Offline export + progress + multithreading (higher throughput, less temporal smoothing):
```powershell
.\.venv\Scripts\python.exe apps/main.py --source input.mp4 --pose heavy --out out.mp4 --no-show --workers 4
```

## Coding Style & Naming Conventions

- Python, 4-space indentation, keep functions small and typed where practical.
- Prefer clear module boundaries: UI code stays in `app_ui.py`, inference/logic stays in `vision_pipeline.py`.
- Avoid committing large artifacts (models/videos). Keep `.gitignore` up to date.

## Testing Guidelines

No formal test suite yet. Minimum checks before opening a PR:
```powershell
.\.venv\Scripts\python.exe -m py_compile .\apps\main.py .\apps\app_ui.py .\core\vision_pipeline.py
```

## Commit & Pull Request Guidelines

- No commit history exists yet; use a simple Conventional Commits style going forward:
  - `feat: ...`, `fix: ...`, `refactor: ...`, `docs: ...`
- PRs should include: what changed, how to run it (CLI/UI commands), and screenshots for UI changes.

## 任务完成规范

- **每次完成任务后，必须将修改总结写入 `change.md` 文件**
- 记录内容应包括：修改日期、问题描述、修改内容、验证方法

