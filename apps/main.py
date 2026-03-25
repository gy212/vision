"""
MediaPipe real-time demo (camera or video) with:
  - PoseLandmarker (pose skeleton)
  - HandLandmarker (hand skeleton) + lightweight V-sign rule

Run (camera):
  .venv\\Scripts\\python.exe main.py

Run (video file):
  .venv\\Scripts\\python.exe main.py --source path\\to\\demo.mp4 --out out.mp4

Quit:
  press "q" in the preview window
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from queue import Queue

import cv2

from core.vision_pipeline import MediaPipePipeline, PipelineConfig


def _open_capture(source: str) -> cv2.VideoCapture:
    # If source is a digit, treat as camera index.
    if source.isdigit():
        idx = int(source)
        # CAP_DSHOW tends to start faster on Windows.
        return cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def _fmt_seconds(s: float) -> str:
    s = max(0.0, float(s))
    m, sec = divmod(int(s + 0.5), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:d}:{sec:02d}"


def _print_progress(done: int, total: int, t0: float) -> None:
    if total <= 0:
        elapsed = time.monotonic() - t0
        rate = done / max(1e-9, elapsed)
        msg = f"processed={done}  {rate:5.1f} fps  elapsed {_fmt_seconds(elapsed)}"
        print("\r" + msg, end="", flush=True)
        return

    frac = min(1.0, max(0.0, done / total))
    width = 28
    filled = int(frac * width)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.monotonic() - t0
    rate = done / max(1e-9, elapsed)
    eta = (total - done) / max(1e-9, rate)
    msg = f"[{bar}] {done}/{total} ({frac*100:5.1f}%)  {rate:5.1f} fps  ETA {_fmt_seconds(eta)}"
    print("\r" + msg, end="", flush=True)
    if done >= total:
        print()


def _process_video_multithread(
    *,
    cap: cv2.VideoCapture,
    out_path: str | None,
    show: bool,
    pose_variant: str,
    workers: int,
) -> None:
    # This mode parallelizes per-frame processing using IMAGE mode pipelines.
    # Trade-off: no temporal tracking/smoothing, but better CPU utilization and throughput.
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_for_ts = src_fps if src_fps > 1e-3 else 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    writer: cv2.VideoWriter | None = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_for_ts, (w, h))

    stop_evt = threading.Event()
    frame_q: "Queue[tuple[int, object] | None]" = Queue(maxsize=workers * 2)
    result_q: "Queue[tuple[int, object, list[str]]]" = Queue(maxsize=workers * 2)

    models_dir = Path(__file__).resolve().parent / "models"

    def reader() -> None:
        idx = 0
        while not stop_evt.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            frame_q.put((idx, frame))
            idx += 1
        for _ in range(workers):
            frame_q.put(None)

    def worker(worker_id: int) -> None:
        pipe = MediaPipePipeline(
            models_dir=models_dir,
            cfg=PipelineConfig(pose_variant=pose_variant, running_mode="image"),
        )
        while True:
            item = frame_q.get()
            if item is None:
                break
            if stop_evt.is_set():
                continue
            idx, frame = item
            annotated, actions = pipe.annotate(frame, timestamp_ms=None)
            result_q.put((idx, annotated, actions))

    t_reader = threading.Thread(target=reader, daemon=True)
    t_reader.start()
    workers_ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(workers)]
    for t in workers_ts:
        t.start()

    next_idx = 0
    pending: dict[int, tuple[object, list[str]]] = {}

    t0 = time.monotonic()
    written = 0
    printed_done = 0

    while True:
        if next_idx in pending:
            annotated, actions = pending.pop(next_idx)
            # Overlay labels + fps
            y = 30
            for a in actions[:5]:
                cv2.putText(annotated, a, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 240), 2, cv2.LINE_AA)
                y += 30
            fps = written / max(1e-6, (time.monotonic() - t0))
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(
                annotated,
                "Press 'q' to quit",
                (10, annotated.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            if writer is not None:
                writer.write(annotated)
                written += 1
                if written != printed_done:
                    _print_progress(written, total, t0)
                    printed_done = written

            if show:
                cv2.imshow("MediaPipe (Video/Camera)", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    stop_evt.set()
                    break

            next_idx += 1
            continue

        # When exporting, wait for results; for preview-only, still need ordering.
        try:
            idx, annotated, actions = result_q.get(timeout=0.2)
        except Exception:
            # All workers finished and buffer drained.
            alive = any(t.is_alive() for t in workers_ts)
            if (not alive) and (not pending):
                break
            continue

        pending[int(idx)] = (annotated, actions)

        # Exit when we've written all frames (if frame count is known).
        if total > 0 and next_idx >= total and not pending:
            break

    stop_evt.set()
    t_reader.join(timeout=2.0)
    for t in workers_ts:
        t.join(timeout=2.0)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def run(source: str, *, show: bool = True, out_path: str | None = None, pose_variant: str = "full") -> None:
    cap = _open_capture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    # Try to request a reasonable camera resolution; for files it is ignored.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    is_file = not source.isdigit()
    fps_for_ts = src_fps if (is_file and src_fps > 1e-3) else 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if is_file else 0

    models_dir = Path(__file__).resolve().parent / "models"
    pipe = MediaPipePipeline(models_dir=models_dir, cfg=PipelineConfig(pose_variant=pose_variant, running_mode="video"))

    writer: cv2.VideoWriter | None = None
    if out_path:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_for_ts, (w, h))

    t0 = time.monotonic()
    frame_count = 0
    printed_done = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_ms = pipe.next_timestamp_ms(is_file=is_file, fps_for_ts=fps_for_ts)
        annotated, actions = pipe.annotate(frame, timestamp_ms=timestamp_ms)

        # Overlay labels
        h, w = annotated.shape[:2]
        y = 30
        for a in actions[:5]:
            cv2.putText(annotated, a, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 240), 2, cv2.LINE_AA)
            y += 30

        frame_count += 1
        fps = frame_count / max(1e-6, (time.monotonic() - t0))
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(annotated, "Press 'q' to quit", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if writer is not None:
            writer.write(annotated)
            if is_file and out_path:
                if frame_count != printed_done and (frame_count % 5 == 0 or (total > 0 and frame_count == total)):
                    _print_progress(frame_count, total, t0)
                    printed_done = frame_count

        if show:
            cv2.imshow("MediaPipe (Video/Camera)", annotated)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="Camera index like 0/1, or a video file path like demo.mp4")
    p.add_argument("--pose", default="full", choices=["lite", "full", "heavy"], help="Pose model variant")
    p.add_argument("--no-show", action="store_true", help="Disable preview window (useful for batch)")
    p.add_argument("--out", default=None, help="Optional output video path, e.g. out.mp4")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Offline video worker threads (>=1). For >1, uses per-frame IMAGE mode (better throughput, less temporal smoothing).",
    )
    args = p.parse_args()

    # Route to multithreaded path only for offline videos.
    if (not args.source.isdigit()) and args.workers and args.workers > 1:
        cap = _open_capture(args.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {args.source}")
        _process_video_multithread(
            cap=cap,
            out_path=args.out,
            show=not args.no_show,
            pose_variant=args.pose,
            workers=max(1, int(args.workers)),
        )
    else:
        run(args.source, show=not args.no_show, out_path=args.out, pose_variant=args.pose)


if __name__ == "__main__":
    main()
