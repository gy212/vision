from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from core.vision_pipeline import MediaPipePipeline, PipelineConfig
from core.pose_features import find_active_range, motion_energy, normalize_pose_xy_v3
from core.video_writer import open_video_writer


#
# NOTE: normalization/segment heuristics live in `pose_features.py` so they can be reused
# by other backends (template matching, UI actions, etc.).
#


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input video file path")
    ap.add_argument("--pose", default="full", choices=["lite", "full", "heavy"], help="Pose model variant")
    ap.add_argument("--out", default=None, help="Output template path (.npz). Default: templates/<stem>_<pose>.npz")
    ap.add_argument("--start", type=int, default=None, help="Override start frame (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="Override end frame (inclusive)")
    ap.add_argument("--preview", action="store_true", help="Export a preview video of the extracted segment")
    args = ap.parse_args()

    video_path = Path(args.video)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    models_dir = Path(__file__).resolve().parent / "models"
    pipe = MediaPipePipeline(
        models_dir=models_dir,
        cfg=PipelineConfig(pose_variant=args.pose, running_mode="video", enable_hands=False),
    )

    feats: list[np.ndarray] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(frame_idx * 1000.0 / fps)
        pose_landmarks, _hands = pipe.infer(frame, timestamp_ms=ts)
        f = normalize_pose_xy_v3(pose_landmarks)
        if f is None:
            # Keep alignment: repeat last valid if available, else zeros.
            if feats:
                f = feats[-1].copy()
            else:
                f = np.zeros((22, 2), dtype=np.float32)
        feats.append(f)
        frame_idx += 1

    cap.release()

    if not feats:
        raise RuntimeError("No frames read from video")

    feat_arr = np.stack(feats, axis=0)  # (T, 22, 2)
    seq = feat_arr.reshape(feat_arr.shape[0], -1)  # (T, 44)
    energy = motion_energy(seq)
    auto_start, auto_end = find_active_range(energy, pad=10)
    start = int(args.start) if args.start is not None else int(auto_start)
    end = int(args.end) if args.end is not None else int(auto_end)
    start = max(0, min(start, feat_arr.shape[0] - 1))
    end = max(start, min(end, feat_arr.shape[0] - 1))

    out_dir = Path(__file__).resolve().parent / "templates"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (out_dir / f"{video_path.stem}_{args.pose}.npz")

    meta = {
        "video": str(video_path),
        "fps": fps,
        "frame_count": n_frames,
        "start_frame": int(start),
        "end_frame": int(end),
        "pose_variant": args.pose,
        "feature_layout": "pose_indices_11_32_xy_rot_scale_norm_v3",
        "running_mode": "video",
        "cfg": asdict(PipelineConfig(pose_variant=args.pose, running_mode="video", enable_hands=False)),
    }

    np.savez_compressed(
        out_path,
        features=feat_arr[start : end + 1],
        meta=np.array(meta, dtype=object),
    )
    print(f"Saved template: {out_path}")
    print(f"Segment frames: {start}..{end} (len={end - start + 1}, fps={fps:.3f})")
    print(f"Auto segment:   {auto_start}..{auto_end}")

    if args.preview:
        preview_path = out_path.with_suffix(".preview.mp4")
        cap2 = cv2.VideoCapture(str(video_path))
        if not cap2.isOpened():
            raise RuntimeError(f"Cannot reopen video: {video_path}")
        w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        vw, actual_path, codec = open_video_writer(preview_path, fps=fps, size=(w, h))

        pipe2 = MediaPipePipeline(
            models_dir=models_dir,
            cfg=PipelineConfig(pose_variant=args.pose, running_mode="video", enable_hands=False),
        )
        i = 0
        while True:
            ok, frame = cap2.read()
            if not ok:
                break
            if i < start:
                i += 1
                continue
            if i > end:
                break
            ts = int(i * 1000.0 / fps)
            annotated, _actions = pipe2.annotate(frame, timestamp_ms=ts)
            vw.write(annotated)
            i += 1

        cap2.release()
        vw.release()
        print(f"Saved preview: {actual_path} (codec={codec})")


if __name__ == "__main__":
    main()
