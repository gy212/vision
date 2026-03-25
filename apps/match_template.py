from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from core.vision_pipeline import MediaPipePipeline, PipelineConfig
from core.pose_features import normalize_pose_xy, normalize_pose_xy_v1, normalize_pose_xy_v3, subsequence_dtw
from core.video_writer import open_video_writer


def _extract_features(
    video_path: Path,
    *,
    pose_variant: str,
    normalizer=normalize_pose_xy,
) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0

    models_dir = Path(__file__).resolve().parent / "models"
    pipe = MediaPipePipeline(
        models_dir=models_dir,
        cfg=PipelineConfig(pose_variant=pose_variant, running_mode="video", enable_hands=False),
    )

    feats: list[np.ndarray] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(i * 1000.0 / fps)
        pose_landmarks, _hands = pipe.infer(frame, timestamp_ms=ts)
        f = normalizer(pose_landmarks)
        if f is None:
            if feats:
                f = feats[-1].copy()
            else:
                f = np.zeros((22, 2), dtype=np.float32)
        feats.append(f)
        i += 1
    cap.release()

    if not feats:
        raise RuntimeError("No frames read")
    return np.stack(feats, axis=0), fps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="Template .npz produced by make_template.py")
    ap.add_argument("--video", required=True, help="Target video to search")
    ap.add_argument("--pose", default=None, choices=["lite", "full", "heavy"], help="Override pose variant")
    ap.add_argument("--preview", action="store_true", help="Export best-match preview video")
    args = ap.parse_args()

    tpl = np.load(args.template, allow_pickle=True)
    query = tpl["features"]
    meta = tpl["meta"].item()
    pose_variant = args.pose or meta.get("pose_variant", "full")
    layout = str(meta.get("feature_layout", "pose_indices_11_32_xy_rot_scale_norm"))
    if layout.endswith("_v3"):
        normalizer = normalize_pose_xy_v3
    elif layout.endswith("_v2"):
        normalizer = normalize_pose_xy
    else:
        normalizer = normalize_pose_xy_v1

    video_path = Path(args.video)
    seq, fps = _extract_features(video_path, pose_variant=pose_variant, normalizer=normalizer)

    cost, start, end = subsequence_dtw(query, seq)
    avg_cost = cost / max(1, query.shape[0])
    # Baseline normalization: score=1.0 when avg_cost=0, score=0.5 when avg_cost=baseline
    baseline = 2.0
    score = float(baseline / (baseline + avg_cost))

    print(f"Template: {args.template}")
    print(f"Video:    {video_path}")
    print(f"Match:    frames {start}..{end}  (t={start/fps:.2f}s..{end/fps:.2f}s)")
    print(f"Cost:     total={cost:.2f}  avg/frame={avg_cost:.3f}  score={score:.3f}")

    if args.preview:
        out_dir = Path(__file__).resolve().parent / "templates"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{video_path.stem}.match_{Path(args.template).stem}.mp4"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot reopen video: {video_path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        vw, actual_path, codec = open_video_writer(out_path, fps=fps, size=(w, h))

        models_dir = Path(__file__).resolve().parent / "models"
        pipe = MediaPipePipeline(
            models_dir=models_dir,
            cfg=PipelineConfig(pose_variant=pose_variant, running_mode="video", enable_hands=False),
        )
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i < start:
                i += 1
                continue
            if i > end:
                break
            ts = int(i * 1000.0 / fps)
            annotated, _actions = pipe.annotate(frame, timestamp_ms=ts)
            vw.write(annotated)
            i += 1
        cap.release()
        vw.release()
        print(f"Saved match preview: {actual_path} (codec={codec})")


if __name__ == "__main__":
    main()
