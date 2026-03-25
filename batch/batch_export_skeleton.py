from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from core.vision_pipeline import MediaPipePipeline, PipelineConfig

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def _iter_videos(root: Path, *, skip_keywords: tuple[str, ...]) -> list[Path]:
    videos: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        if _should_skip(p, skip_keywords=skip_keywords):
            continue
        videos.append(p)
    return sorted(videos)


def _should_skip(path: Path, *, skip_keywords: tuple[str, ...]) -> bool:
    if not skip_keywords:
        return False
    for part in path.parts:
        for kw in skip_keywords:
            if kw and kw in part:
                return True
    return False


def _sanitize_name(name: str) -> str:
    cleaned = name.strip().rstrip(".")
    for ch in "\\/:*?\"<>|":
        cleaned = cleaned.replace(ch, "_")
    return cleaned or "video"


def _unique_name(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    idx = 2
    while True:
        cand = f"{base}_{idx}"
        if cand not in used:
            used.add(cand)
            return cand
        idx += 1


def _extract_pose_and_video(
    video_path: Path,
    *,
    out_video: Path | None,
    pose_variant: str,
    draw_face: bool,
) -> tuple[np.ndarray, dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    models_dir = Path(__file__).resolve().parent / "models"
    pipe = MediaPipePipeline(
        models_dir=models_dir,
        cfg=PipelineConfig(
            pose_variant=pose_variant,
            running_mode="video",
            enable_hands=False,
            draw_pose_face=draw_face,
        ),
    )

    writer: cv2.VideoWriter | None = None
    if out_video is not None:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))

    out: list[np.ndarray] = []
    last: np.ndarray | None = None

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(i * 1000.0 / fps)
        pose_landmarks, _hands = pipe.infer(frame, timestamp_ms=ts)
        if pose_landmarks is None:
            arr = last.copy() if last is not None else np.zeros((33, 4), dtype=np.float32)
        else:
            arr = np.zeros((33, 4), dtype=np.float32)
            for j in range(min(33, len(pose_landmarks))):
                lm = pose_landmarks[j]
                arr[j, 0] = float(getattr(lm, "x", 0.0))
                arr[j, 1] = float(getattr(lm, "y", 0.0))
                arr[j, 2] = float(getattr(lm, "z", 0.0))
                arr[j, 3] = float(getattr(lm, "visibility", 0.0))
            last = arr

        out.append(arr)

        if writer is not None:
            annotated = frame.copy()
            MediaPipePipeline._draw_pose(annotated, pose_landmarks, w, h, draw_face=draw_face)
            writer.write(annotated)

        i += 1

    cap.release()
    if writer is not None:
        writer.release()

    if not out:
        raise RuntimeError(f"未能从视频提取姿态：{video_path}")

    meta = {
        "video": str(video_path),
        "name": video_path.stem.strip(),
        "fps": float(fps),
        "frame_count": int(n_frames),
        "width": int(w),
        "height": int(h),
        "pose_variant": str(pose_variant),
        "landmark_layout": "pose33_normalized_xyzw(visibility)",
        "skeleton_video": None if out_video is None else str(out_video),
    }
    return np.stack(out, axis=0), meta


def _write_manifest(rows: Iterable[dict], out_path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch export Pose33 skeleton data + skeleton videos.")
    ap.add_argument("--source_dir", default="标准动作视频--分解版", help="Input root folder")
    ap.add_argument("--out_dir", default=None, help="Output root folder (default: outputs/<name>_<ts>)")
    ap.add_argument("--pose", default="full", choices=["lite", "full", "heavy"], help="Pose model variant")
    ap.add_argument(
        "--skip_keywords",
        default="汇总",
        help="Comma-separated keywords to skip folders (default: 汇总)",
    )
    ap.add_argument("--no_video", action="store_true", help="Only export .npz (skip skeleton video)")
    ap.add_argument("--draw_face", action="store_true", help="Draw face landmarks on skeleton video")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir not found: {source_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (Path(__file__).resolve().parent / "outputs" / f"标准动作视频--分解版_骨架_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    skip_keywords = tuple(k.strip() for k in str(args.skip_keywords).split(",") if k.strip())
    videos = _iter_videos(source_dir, skip_keywords=skip_keywords)
    if not videos:
        raise FileNotFoundError(f"No videos found in: {source_dir}")

    used_names_by_dir: dict[Path, set[str]] = {}
    manifest_rows: list[dict] = []

    for idx, video_path in enumerate(videos, start=1):
        rel_parent = video_path.parent.relative_to(source_dir)
        out_subdir = out_dir / rel_parent
        used = used_names_by_dir.setdefault(out_subdir, set())

        base_name = _sanitize_name(video_path.stem)
        safe_name = _unique_name(base_name, used)

        npz_path = out_subdir / f"{safe_name}_pose33_{args.pose}.npz"
        skel_video_path = None if args.no_video else (out_subdir / f"{safe_name}_skeleton_{args.pose}.mp4")

        if not args.overwrite:
            if npz_path.exists() and (skel_video_path is None or skel_video_path.exists()):
                print(f"[{idx}/{len(videos)}] Skip (exists): {video_path}")
                manifest_rows.append(
                    {
                        "action": str(rel_parent),
                        "video_name": video_path.name,
                        "source_video": str(video_path),
                        "skeleton_npz": str(npz_path),
                        "skeleton_video": "" if skel_video_path is None else str(skel_video_path),
                        "pose_variant": str(args.pose),
                    }
                )
                continue

        out_subdir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(videos)}] Processing: {video_path}")
        landmarks, meta = _extract_pose_and_video(
            video_path,
            out_video=skel_video_path,
            pose_variant=str(args.pose),
            draw_face=bool(args.draw_face),
        )
        meta["output_npz"] = str(npz_path)
        meta["action"] = str(rel_parent)

        np.savez_compressed(npz_path, landmarks=landmarks, meta=np.array(meta, dtype=object))

        manifest_rows.append(
            {
                "action": str(rel_parent),
                "video_name": video_path.name,
                "source_video": str(video_path),
                "skeleton_npz": str(npz_path),
                "skeleton_video": "" if skel_video_path is None else str(skel_video_path),
                "pose_variant": str(args.pose),
                "fps": float(meta.get("fps", 0.0)),
                "frame_count": int(meta.get("frame_count", 0)),
                "width": int(meta.get("width", 0)),
                "height": int(meta.get("height", 0)),
            }
        )

    manifest_path = out_dir / "manifest.csv"
    _write_manifest(manifest_rows, manifest_path)
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved skeleton data: {out_dir}")


if __name__ == "__main__":
    main()
