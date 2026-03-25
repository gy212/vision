from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from typing import Callable

import cv2
import numpy as np

from .pose_features import (
    find_active_range,
    mirror_pose_features,
    motion_energy,
    normalize_pose_xy,
    normalize_pose_xy_v1,
    normalize_pose_xy_v3,
    pose_view_score,
    subsequence_dtw,
    subsequence_dtw_with_path,
)
from .rule_scoring import RuleViolation, extract_pose_raw, score_rules
from .vision_pipeline import MediaPipePipeline, PipelineConfig
from .video_writer import open_video_writer


ProgressCb = Callable[[str, int, int], None]  # (stage, done, total)


@dataclass(frozen=True)
class CompareResult:
    template_path: Path
    video_path: Path
    pose_variant: str
    fps: float
    start_frame: int
    end_frame: int
    cost: float
    avg_cost: float
    score: float
    preview_path: Path | None = None
    workers_used: int | None = None


@dataclass(frozen=True)
class RepetitionMatch:
    start_frame: int
    end_frame: int
    avg_cost: float
    score: float


@dataclass(frozen=True)
class JointErrorStat:
    joint: str
    mean_dist: float | None
    p90_dist: float | None
    max_dist: float | None
    valid_frames: int


@dataclass(frozen=True)
class DualCompareResult:
    front_template_path: Path
    side_template_path: Path
    video_path: Path
    pose_variant: str
    fps: float
    front_score: float
    side_score: float
    combined_score: float
    combined_percent: int
    front_matches: tuple[RepetitionMatch, ...]
    side_matches: tuple[RepetitionMatch, ...]
    front_segment: tuple[int, int] | None = None  # (start,end) in original video frames
    side_segment: tuple[int, int] | None = None  # (start,end) in original video frames
    front_rule_score: int | None = None
    side_rule_score: int | None = None
    front_rule_deduction: int | None = None
    side_rule_deduction: int | None = None
    front_rule_violations: tuple[RuleViolation, ...] | None = None
    side_rule_violations: tuple[RuleViolation, ...] | None = None
    front_joint_errors: tuple[JointErrorStat, ...] | None = None
    side_joint_errors: tuple[JointErrorStat, ...] | None = None


def _extract_pose_features(
    video_path: Path,
    *,
    pose_variant: str,
    workers: int = 1,
    normalizer: Callable = normalize_pose_xy,
    compute_view: bool = False,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """
    Extract normalized pose features for the entire video.
    Returns (features[T,22,2], fps, view_scores[T] or None).
    """
    stop_evt = stop_evt or Event()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    workers = max(1, int(workers))

    def _emit(done: int) -> None:
        if progress_cb is not None:
            progress_cb("提取骨架特征", done, total)

    # Single-thread path: VIDEO mode (more stable landmarks).
    if workers == 1:
        models_dir = Path(__file__).resolve().parent / "models"
        pipe = MediaPipePipeline(
            models_dir=models_dir,
            cfg=PipelineConfig(pose_variant=pose_variant, running_mode="video", enable_hands=False),
        )
        feats: list[np.ndarray] = []
        views: list[float] | None = [] if compute_view else None
        last_view = 0.0
        i = 0
        while not stop_evt.is_set():
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
            if views is not None:
                v = pose_view_score(pose_landmarks)
                if v is None:
                    v = last_view
                else:
                    last_view = float(v)
                views.append(float(v))
            i += 1
            if (i % 10 == 0) or (total > 0 and i == total):
                _emit(i)
        cap.release()
        if not feats:
            raise RuntimeError("视频为空或无法读取帧")
        view_arr = np.array(views, dtype=np.float32) if views is not None else None
        return np.stack(feats, axis=0), fps, view_arr

    # Multi-thread path: split the video into segments and process each segment sequentially
    # with VIDEO mode (tracking enabled within each segment). This keeps features comparable
    # to the template (VIDEO mode) and avoids the huge mismatch caused by IMAGE mode.
    if total <= 0:
        # Can't segment without a known frame count; fall back to single-thread.
        cap.release()
        return _extract_pose_features(
            video_path,
            pose_variant=pose_variant,
            workers=1,
            normalizer=normalizer,
            compute_view=compute_view,
            progress_cb=progress_cb,
            stop_evt=stop_evt,
        )

    cap.release()

    models_dir = Path(__file__).resolve().parent / "models"
    feat_arr = np.zeros((total, 22, 2), dtype=np.float32)
    view_arr = np.zeros((total,), dtype=np.float32) if compute_view else None

    done_lock = Lock()
    done = 0

    def _inc_done(n: int = 1) -> None:
        nonlocal done
        with done_lock:
            done += n
            d = done
        if (d % 10 == 0) or d == total:
            _emit(d)

    overlap = 15  # warm-up frames per segment to stabilize tracking

    def seg_worker(seg_start: int, seg_end: int) -> None:
        warm_start = max(0, seg_start - overlap)
        cap2 = cv2.VideoCapture(str(video_path))
        if not cap2.isOpened():
            stop_evt.set()
            return
        # Seek to warm_start; if seek fails, OpenCV will usually continue from 0.
        cap2.set(cv2.CAP_PROP_POS_FRAMES, float(warm_start))

        pipe = MediaPipePipeline(
            models_dir=models_dir,
            cfg=PipelineConfig(pose_variant=pose_variant, running_mode="video", enable_hands=False),
        )

        last: np.ndarray | None = None
        last_view: float | None = None
        i = warm_start
        while (not stop_evt.is_set()) and i <= seg_end:
            ok, frame = cap2.read()
            if not ok:
                break
            ts = int(i * 1000.0 / fps)
            pose_landmarks, _hands = pipe.infer(frame, timestamp_ms=ts)
            f = normalizer(pose_landmarks)
            if f is None:
                f = last.copy() if last is not None else np.zeros((22, 2), dtype=np.float32)
            else:
                last = f

            if i >= seg_start:
                feat_arr[i] = f
                if view_arr is not None:
                    v = pose_view_score(pose_landmarks)
                    if v is None:
                        v = float(last_view or 0.0)
                    else:
                        last_view = float(v)
                    view_arr[i] = float(v)
                _inc_done(1)
            i += 1

        cap2.release()

    # Partition [0, total) across workers
    segs: list[tuple[int, int]] = []
    for k in range(workers):
        s = (k * total) // workers
        e = ((k + 1) * total) // workers - 1
        if e >= s:
            segs.append((s, e))

    threads = [Thread(target=seg_worker, args=segs[i], daemon=True) for i in range(len(segs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if done == 0:
        raise RuntimeError("视频为空或无法读取帧")
    return feat_arr, fps, view_arr


def _smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or x.size == 0:
        return x.astype(np.float32, copy=True)
    k = int(k)
    if k % 2 == 0:
        k += 1
    if x.size < k:
        return x.astype(np.float32, copy=True)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x.astype(np.float32), kernel, mode="same").astype(np.float32)


def _longest_segment(mask: np.ndarray) -> tuple[int, int] | None:
    """Return the longest contiguous True segment (start,end) inclusive."""
    if mask.size == 0:
        return None

    best_s: int | None = None
    best_e: int | None = None
    i = 0
    n = int(mask.size)
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        s = i
        i += 1
        while i < n and bool(mask[i]):
            i += 1
        e = i - 1
        if best_s is None or (e - s) > (best_e - best_s):  # type: ignore[operator]
            best_s, best_e = s, e
    if best_s is None or best_e is None:
        return None
    return int(best_s), int(best_e)


def _trimmed_mean(scores: list[float]) -> float:
    if not scores:
        return 0.0
    if len(scores) >= 3:
        xs = sorted(float(s) for s in scores)
        xs = xs[1:-1]  # drop min/max
        return float(sum(xs) / max(1, len(xs)))
    return float(sum(scores) / len(scores))


def _estimate_period_frames(energy: np.ndarray, fps: float) -> int | None:
    """
    Estimate dominant repetition period (in frames) via normalized autocorrelation on motion energy.
    Returns None if not confident.
    """
    e = energy.astype(np.float32).ravel()
    if e.size < 30:
        return None

    x = e - float(np.mean(e))
    x_std = float(np.std(x))
    if not np.isfinite(x_std) or x_std < 1e-6:
        return None

    fps = float(fps or 0.0) or 30.0
    min_lag = max(6, int(0.4 * fps))
    max_lag = min(int(3.0 * fps), int(e.size // 2))
    if max_lag <= min_lag:
        return None

    best_lag = None
    best_corr = -1.0
    for lag in range(min_lag, max_lag + 1):
        a = x[:-lag]
        b = x[lag:]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        c = float(np.dot(a, b) / denom)
        if c > best_corr:
            best_corr = c
            best_lag = lag

    # Weak periodicity => don't force a cycle cut.
    if best_lag is None or best_corr < 0.15:
        return None
    return int(best_lag)


def _select_representative_cycle(features: np.ndarray, fps: float) -> np.ndarray:
    """
    For a repeated standard video, pick a single representative repetition as the matching query.
    If we can't estimate a stable period, return the original features.
    """
    if features.ndim != 3 or features.shape[1:] != (22, 2):
        return features
    if features.shape[0] < 30:
        return features

    seq = features.reshape(features.shape[0], -1)
    energy = motion_energy(seq)
    period = _estimate_period_frames(energy, fps=float(fps))
    if period is None or period < 8:
        return features

    t = int(features.shape[0])
    mid = t // 2
    start = max(0, mid - (period // 2))

    # Try to snap start/end to low-motion frames to avoid cutting mid-gesture.
    e_frame = np.concatenate([energy[:1], energy]).astype(np.float32)  # len T
    low_thr = float(np.percentile(e_frame, 30))
    low = np.where(e_frame <= low_thr)[0].astype(int)

    def nearest_low(idx: int) -> int | None:
        if low.size == 0:
            return None
        radius = max(3, period // 3)
        lo = int(max(0, idx - radius))
        hi = int(min(t - 1, idx + radius))
        cand = low[(low >= lo) & (low <= hi)]
        if cand.size == 0:
            return None
        j = int(cand[np.argmin(np.abs(cand - idx))])
        return j

    s2 = nearest_low(start)
    if s2 is not None:
        start = int(s2)

    end = min(t, start + period)
    e2 = nearest_low(end)
    if e2 is not None and int(e2) > start + 10:
        end = int(e2)

    if end - start < max(12, period // 2):
        end = min(t, start + period)

    return features[start:end]


def _multi_subsequence_matches(
    query: np.ndarray,
    seq: np.ndarray,
    *,
    baseline: float = 2.0,
    max_matches: int = 30,
    exclusion: int = 5,
    offset: int = 0,
) -> list[RepetitionMatch]:
    """
    Greedy multi-match: repeatedly find the best subsequence DTW match, exclude it, and repeat.
    Returns matches in arbitrary order (not necessarily chronological).
    """
    if query.size == 0 or seq.size == 0:
        return []

    q = query
    q_m = mirror_pose_features(query)
    q_len = int(q.shape[0])
    if q_len <= 0:
        return []

    remaining: list[tuple[int, int]] = [(0, int(seq.shape[0] - 1))]
    out: list[RepetitionMatch] = []

    ref_avg: float | None = None

    while remaining and len(out) < int(max_matches):
        best: tuple[float, int, int] | None = None  # (avg_cost, start, end) in seq-local indices

        for seg_s, seg_e in remaining:
            if seg_e - seg_s + 1 < max(8, q_len // 2):
                continue
            sub = seq[seg_s : seg_e + 1]
            cost1, s1, e1 = subsequence_dtw(q, sub)
            cost2, s2, e2 = subsequence_dtw(q_m, sub)
            avg1 = float(cost1 / max(1, q_len))
            avg2 = float(cost2 / max(1, q_len))
            if avg2 < avg1:
                avg, s, e = avg2, int(s2), int(e2)
            else:
                avg, s, e = avg1, int(s1), int(e1)
            s += int(seg_s)
            e += int(seg_s)

            if best is None or avg < best[0]:
                best = (avg, s, e)

        if best is None:
            break

        avg, s, e = best

        # Stop once matching becomes much worse than the best (avoid collecting random matches).
        if ref_avg is None:
            ref_avg = float(avg)
        else:
            if float(avg) > float(ref_avg) * 2.5 + 0.5:
                break

        score = float(baseline / (baseline + float(avg)))
        out.append(
            RepetitionMatch(
                start_frame=int(offset + s),
                end_frame=int(offset + e),
                avg_cost=float(avg),
                score=float(score),
            )
        )

        # Exclude matched region (with padding) and keep remaining segments.
        exc_s = max(0, int(s - exclusion))
        exc_e = min(int(seq.shape[0] - 1), int(e + exclusion))
        new_remaining: list[tuple[int, int]] = []
        for seg_s, seg_e in remaining:
            if exc_e < seg_s or exc_s > seg_e:
                new_remaining.append((seg_s, seg_e))
                continue
            if seg_s < exc_s:
                new_remaining.append((seg_s, exc_s - 1))
            if exc_e < seg_e:
                new_remaining.append((exc_e + 1, seg_e))
        remaining = new_remaining

    return out


def create_template_from_video(
    video_path: str | Path,
    *,
    pose_variant: str = "heavy",
    start: int | None = None,
    end: int | None = None,
    out_path: str | Path | None = None,
    workers: int = 1,
    preview: bool = False,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> Path:
    video_path = Path(video_path)
    features, fps, _view = _extract_pose_features(
        video_path,
        pose_variant=pose_variant,
        workers=workers,
        normalizer=normalize_pose_xy_v3,
        progress_cb=progress_cb,
        stop_evt=stop_evt,
    )

    seq = features.reshape(features.shape[0], -1)
    energy = motion_energy(seq)
    auto_start, auto_end = find_active_range(energy, pad=10)

    start_i = int(start) if start is not None else int(auto_start)
    end_i = int(end) if end is not None else int(auto_end)
    start_i = max(0, min(start_i, features.shape[0] - 1))
    end_i = max(start_i, min(end_i, features.shape[0] - 1))

    out_dir = Path(__file__).resolve().parent / "templates"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path) if out_path else (out_dir / f"{video_path.stem}_{pose_variant}.npz")

    meta = {
        "video": str(video_path),
        "fps": float(fps),
        "frame_count": int(features.shape[0]),
        "start_frame": int(start_i),
        "end_frame": int(end_i),
        "auto_start_frame": int(auto_start),
        "auto_end_frame": int(auto_end),
        "pose_variant": pose_variant,
        "feature_layout": "pose_indices_11_32_xy_rot_scale_norm_v3",
        "running_mode": "video",
        "cfg": asdict(PipelineConfig(pose_variant=pose_variant, running_mode="video", enable_hands=False)),
    }

    np.savez_compressed(
        out_path,
        features=features[start_i : end_i + 1],
        meta=np.array(meta, dtype=object),
    )

    if preview and progress_cb is not None:
        progress_cb("导出模板预览", 0, 1)
    if preview:
        preview_path = out_path.with_suffix(".preview.mp4")
        export_match_preview(
            video_path,
            out_path=preview_path,
            start_frame=start_i,
            end_frame=end_i,
            pose_variant=pose_variant,
            progress_cb=progress_cb,
            stop_evt=stop_evt,
        )

    return out_path


def compare_video_to_template(
    template_path: str | Path,
    video_path: str | Path,
    *,
    pose_variant: str | None = None,
    workers: int = 1,
    preview_out: str | Path | None = None,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> CompareResult:
    stop_evt = stop_evt or Event()
    template_path = Path(template_path)
    video_path = Path(video_path)

    tpl = np.load(template_path, allow_pickle=True)
    query = tpl["features"]
    meta = tpl["meta"].item()
    pv = pose_variant or meta.get("pose_variant", "full")
    layout = str(meta.get("feature_layout", "pose_indices_11_32_xy_rot_scale_norm"))
    if layout.endswith("_v3"):
        normalizer = normalize_pose_xy_v3
    elif layout.endswith("_v2"):
        normalizer = normalize_pose_xy
    else:
        normalizer = normalize_pose_xy_v1
    tpl_mode = str(meta.get("running_mode") or (meta.get("cfg") or {}).get("running_mode") or "video").lower()
    workers_eff = 1 if (tpl_mode == "video" and int(workers) > 1) else int(workers)
    workers_eff = max(1, workers_eff)

    seq, fps, _view = _extract_pose_features(
        video_path,
        pose_variant=pv,
        workers=workers_eff,
        normalizer=normalizer,
        progress_cb=progress_cb,
        stop_evt=stop_evt,
    )

    if progress_cb is not None:
        progress_cb("计算相似度", 0, 1)
    t0 = time.monotonic()
    cost, start, end = subsequence_dtw(query, seq)
    avg_cost = cost / max(1, int(query.shape[0]))
    # Baseline normalization: score=1.0 when avg_cost=0, score=0.5 when avg_cost=baseline
    baseline = 2.0
    score = float(baseline / (baseline + avg_cost))
    _ = time.monotonic() - t0

    actual_preview_path: Path | None = None
    if preview_out:
        actual_preview_path = export_match_preview(
            video_path,
            out_path=Path(preview_out),
            start_frame=int(start),
            end_frame=int(end),
            pose_variant=pv,
            progress_cb=progress_cb,
            stop_evt=stop_evt,
        )

    return CompareResult(
        template_path=template_path,
        video_path=video_path,
        pose_variant=pv,
        fps=float(fps),
        start_frame=int(start),
        end_frame=int(end),
        cost=float(cost),
        avg_cost=float(avg_cost),
        score=float(score),
        preview_path=actual_preview_path,
        workers_used=workers_eff,
    )


def compare_video_to_dual_templates(
    front_template_path: str | Path,
    side_template_path: str | Path,
    video_path: str | Path,
    *,
    pose_variant: str | None = None,
    workers: int = 1,
    w_front: float = 0.4,
    w_side: float = 0.6,
    baseline: float = 2.0,
    enable_rules: bool = False,
    action_scope: str = "both",
    enable_error_analysis: bool = False,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> DualCompareResult:
    """
    Compare a student's single long video (contains both front+side view) against two standard templates.

    - Auto-split student video into front/side segments using pose "frontness" score.
    - For each view, auto-pick a single repetition from the standard template as the DTW query.
    - Run greedy multi-match subsequence DTW to score multiple repetitions.
    - Aggregate by trimmed mean: drop max & min, then average (when >=3 reps).
    - Optional: rule-based scoring on raw Pose33 (enable_rules/action_scope).

    Returns both per-view scores (0..1) and combined integer percent (0..100).
    """
    stop_evt = stop_evt or Event()
    front_template_path = Path(front_template_path)
    side_template_path = Path(side_template_path)
    video_path = Path(video_path)
    action_scope = str(action_scope or "both").lower()

    tpl_f = np.load(front_template_path, allow_pickle=True)
    tpl_s = np.load(side_template_path, allow_pickle=True)

    feat_f = tpl_f["features"]
    feat_s = tpl_s["features"]
    meta_f = tpl_f["meta"].item()
    meta_s = tpl_s["meta"].item()

    layout_f = str(meta_f.get("feature_layout", "pose_indices_11_32_xy_rot_scale_norm"))
    layout_s = str(meta_s.get("feature_layout", "pose_indices_11_32_xy_rot_scale_norm"))
    def _layout_ver(layout: str) -> str:
        if layout.endswith("_v3"):
            return "v3"
        if layout.endswith("_v2"):
            return "v2"
        return "v1"

    if _layout_ver(layout_f) != _layout_ver(layout_s):
        raise ValueError("Front/side templates use different feature layouts; regenerate templates with the same version.")

    if layout_f.endswith("_v3"):
        normalizer = normalize_pose_xy_v3
    elif layout_f.endswith("_v2"):
        normalizer = normalize_pose_xy
    else:
        normalizer = normalize_pose_xy_v1
    pv = pose_variant or meta_f.get("pose_variant") or meta_s.get("pose_variant") or "full"

    tpl_mode_f = str(meta_f.get("running_mode") or (meta_f.get("cfg") or {}).get("running_mode") or "video").lower()
    tpl_mode_s = str(meta_s.get("running_mode") or (meta_s.get("cfg") or {}).get("running_mode") or "video").lower()
    tpl_video_mode = (tpl_mode_f == "video") or (tpl_mode_s == "video")
    workers_eff = 1 if (tpl_video_mode and int(workers) > 1) else int(workers)
    workers_eff = max(1, workers_eff)

    seq, fps, view_scores = _extract_pose_features(
        video_path,
        pose_variant=pv,
        workers=workers_eff,
        normalizer=normalizer,
        compute_view=True,
        progress_cb=progress_cb,
        stop_evt=stop_evt,
    )

    # 双模板比对核心流程：
    # A) 正/侧视角切分；B) DTW 多次匹配与聚合；C) 可选规则扣分与明细输出。

    # 1) 按“正面程度”把长视频拆成正/侧两段（无需手动标注转身点）。
    #    正面程度来自肩宽/躯干长度 + 左右可见度平衡，先平滑再二分。
    front_seg: tuple[int, int] | None = None
    side_seg: tuple[int, int] | None = None
    if view_scores is not None and view_scores.size == seq.shape[0] and seq.shape[0] >= 10:
        vs = np.nan_to_num(view_scores.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        k = int(max(5, min(51, round(float(fps) * 0.5))))
        vs_s = _smooth_1d(vs, k=k)
        thr = float(np.median(vs_s))
        front_mask = vs_s >= thr
        front_seg = _longest_segment(front_mask)
        side_seg = _longest_segment(~front_mask)

        # 去掉临界帧：转身附近往往更模糊，缩短边缘以提升稳定性。
        margin = int(max(5, min(20, round(float(fps) * 0.3))))

        def shrink(seg: tuple[int, int] | None) -> tuple[int, int] | None:
            if seg is None:
                return None
            s, e = int(seg[0]), int(seg[1])
            if e - s + 1 > (2 * margin + 10):
                s += margin
                e -= margin
            if e <= s:
                return None
            return s, e

        front_seg = shrink(front_seg)
        side_seg = shrink(side_seg)

    # 1.5) 可选规则评分：基于原始 Pose33，分视角段独立评估。
    #      规则采用“违规帧占比 >= 阈值则扣分”，降低单帧抖动影响。
    rule_front = None
    rule_side = None
    if enable_rules:
        def _score_rules_for_seg(seg: tuple[int, int] | None, *, view: str):
            if seg is None:
                s = e = None
            else:
                s, e = int(seg[0]), int(seg[1])
            raw, _meta = extract_pose_raw(
                video_path,
                pose_variant=pv,
                start_frame=s,
                end_frame=e,
            )
            return score_rules(raw, view=view, action_scope=action_scope)

        rule_front = _score_rules_for_seg(front_seg, view="front")
        rule_side = _score_rules_for_seg(side_seg, view="side")

    joint_names_11_32 = [
        "L_SHOULDER",
        "R_SHOULDER",
        "L_ELBOW",
        "R_ELBOW",
        "L_WRIST",
        "R_WRIST",
        "L_PINKY",
        "R_PINKY",
        "L_INDEX",
        "R_INDEX",
        "L_THUMB",
        "R_THUMB",
        "L_HIP",
        "R_HIP",
        "L_KNEE",
        "R_KNEE",
        "L_ANKLE",
        "R_ANKLE",
        "L_HEEL",
        "R_HEEL",
        "L_FOOT_INDEX",
        "R_FOOT_INDEX",
    ]

    def _swap_lr(name: str) -> str:
        if name.startswith("L_"):
            return "R_" + name[2:]
        if name.startswith("R_"):
            return "L_" + name[2:]
        return name

    def score_view(
        tpl_features: np.ndarray,
        tpl_meta: dict,
        seg: tuple[int, int] | None,
    ) -> tuple[float, list[RepetitionMatch], list[JointErrorStat] | None]:
        # 2) 从标准模板中挑一个代表周期，避免整段重复导致匹配过长/过慢。
        tpl_fps = float(tpl_meta.get("fps") or fps)
        query = _select_representative_cycle(tpl_features, fps=tpl_fps)
        if query.size == 0:
            return 0.0, [], None

        # 3) 选择学员分段（缺失则回退整段视频）。
        if seg is None:
            seg_s, seg_e = 0, int(seq.shape[0] - 1)
        else:
            seg_s, seg_e = int(seg[0]), int(seg[1])
            seg_s = max(0, min(seg_s, int(seq.shape[0] - 1)))
            seg_e = max(seg_s, min(seg_e, int(seq.shape[0] - 1)))

        seg_seq = seq[seg_s : seg_e + 1]
        seg_offset = seg_s

        # 4) 限制到更“活跃”的区间，减少静止帧对匹配的干扰。
        if seg_seq.shape[0] >= 2:
            energy = motion_energy(seg_seq.reshape(seg_seq.shape[0], -1))
            a_s, a_e = find_active_range(energy, pad=10)
            a_s = max(0, min(int(a_s), int(seg_seq.shape[0] - 1)))
            a_e = max(a_s, min(int(a_e), int(seg_seq.shape[0] - 1)))
            seg_seq = seg_seq[a_s : a_e + 1]
            seg_offset += int(a_s)

        exclusion = max(3, int(round(0.2 * float(query.shape[0]))))
        max_matches = int(min(30, max(1, round(float(seg_seq.shape[0]) / max(1.0, float(query.shape[0]))))))

        matches = _multi_subsequence_matches(
            query,
            seg_seq,
            baseline=float(baseline),
            max_matches=max_matches,
            exclusion=exclusion,
            offset=seg_offset,
        )

        # 兜底：即便没有找到重复匹配，也保证有一个分数输出。
        if not matches and seg_seq.size > 0:
            cost, s, e = subsequence_dtw(query, seg_seq)
            avg = float(cost / max(1, int(query.shape[0])))
            matches = [
                RepetitionMatch(
                    start_frame=int(seg_offset + s),
                    end_frame=int(seg_offset + e),
                    avg_cost=float(avg),
                    score=float(float(baseline) / (float(baseline) + float(avg))),
                )
            ]

        scores = [m.score for m in matches]
        joint_stats: list[JointErrorStat] | None = None
        if enable_error_analysis and seg_seq.size > 0 and query.size > 0:
            q_m = mirror_pose_features(query)
            cost1, s1, e1, path1 = subsequence_dtw_with_path(query, seg_seq)
            cost2, s2, e2, path2 = subsequence_dtw_with_path(q_m, seg_seq)
            if float(cost2) < float(cost1):
                q_eff = q_m
                start_i = int(s2)
                end_i = int(e2)
                path = path2
                mirrored = True
            else:
                q_eff = query
                start_i = int(s1)
                end_i = int(e1)
                path = path1
                mirrored = False

            if start_i <= end_i and path:
                raw, _meta = extract_pose_raw(
                    video_path,
                    pose_variant=pv,
                    start_frame=int(seg_offset),
                    end_frame=int(seg_offset + int(seg_seq.shape[0]) - 1),
                )
                thr_vis = 0.5
                dists: list[list[float]] = [[] for _ in range(22)]
                for qi, sj in path:
                    qi_i = int(qi)
                    sj_i = int(sj)
                    if qi_i < 0 or qi_i >= int(q_eff.shape[0]) or sj_i < 0 or sj_i >= int(seg_seq.shape[0]):
                        continue
                    vis = raw[sj_i, 11:33, 3].astype(np.float32)
                    a = q_eff[qi_i].astype(np.float32)
                    b = seg_seq[sj_i].astype(np.float32)
                    for k in range(22):
                        if float(vis[k]) < thr_vis:
                            continue
                        d = float(np.linalg.norm(a[k] - b[k]))
                        if np.isfinite(d):
                            dists[k].append(d)

                names = joint_names_11_32 if not mirrored else [_swap_lr(n) for n in joint_names_11_32]
                joint_stats = []
                for k in range(22):
                    arr = np.array(dists[k], dtype=np.float32)
                    if arr.size == 0:
                        joint_stats.append(
                            JointErrorStat(joint=str(names[k]), mean_dist=None, p90_dist=None, max_dist=None, valid_frames=0)
                        )
                    else:
                        joint_stats.append(
                            JointErrorStat(
                                joint=str(names[k]),
                                mean_dist=float(np.mean(arr)),
                                p90_dist=float(np.percentile(arr, 90)),
                                max_dist=float(np.max(arr)),
                                valid_frames=int(arr.size),
                            )
                        )

        return _trimmed_mean(scores), matches, joint_stats

    front_score, front_matches, front_joint_errors = score_view(feat_f, meta_f, front_seg)
    side_score, side_matches, side_joint_errors = score_view(feat_s, meta_s, side_seg)

    combined = float((w_front * float(front_score)) + (w_side * float(side_score)))
    combined = float(np.clip(combined, 0.0, 1.0))
    pct = int(np.clip(int(round(combined * 100.0)), 0, 100))

    return DualCompareResult(
        front_template_path=front_template_path,
        side_template_path=side_template_path,
        video_path=video_path,
        pose_variant=pv,
        fps=float(fps),
        front_score=float(front_score),
        side_score=float(side_score),
        combined_score=float(combined),
        combined_percent=int(pct),
        front_matches=tuple(front_matches),
        side_matches=tuple(side_matches),
        front_segment=front_seg,
        side_segment=side_seg,
        front_rule_score=None if rule_front is None else int(rule_front.score),
        side_rule_score=None if rule_side is None else int(rule_side.score),
        front_rule_deduction=None if rule_front is None else int(rule_front.total_deduction),
        side_rule_deduction=None if rule_side is None else int(rule_side.total_deduction),
        front_rule_violations=None if rule_front is None else tuple(rule_front.violations),
        side_rule_violations=None if rule_side is None else tuple(rule_side.violations),
        front_joint_errors=None if front_joint_errors is None else tuple(front_joint_errors),
        side_joint_errors=None if side_joint_errors is None else tuple(side_joint_errors),
    )


def export_match_preview(
    video_path: Path,
    *,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    pose_variant: str,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> Path:
    stop_evt = stop_evt or Event()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    vw, actual_path, codec = open_video_writer(out_path, fps=fps, size=(w, h))

    models_dir = Path(__file__).resolve().parent / "models"
    pipe = MediaPipePipeline(
        models_dir=models_dir,
        cfg=PipelineConfig(pose_variant=pose_variant, running_mode="video", enable_hands=False),
    )

    total = max(0, int(end_frame - start_frame + 1))
    done = 0
    i = 0
    while not stop_evt.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        if i < start_frame:
            i += 1
            continue
        if i > end_frame:
            break
        ts = int(i * 1000.0 / fps)
        annotated, _actions = pipe.annotate(frame, timestamp_ms=ts)
        vw.write(annotated)
        i += 1
        done += 1
        if progress_cb is not None and (done % 5 == 0 or done == total):
            progress_cb("导出匹配预览", done, total)

    cap.release()
    vw.release()
    return actual_path
