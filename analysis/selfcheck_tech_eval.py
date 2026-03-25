from __future__ import annotations

import re

import numpy as np

from analysis.tech_eval import (
    L_ANKLE,
    L_ELBOW,
    L_FOOT_INDEX,
    L_HEEL,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_ANKLE,
    R_ELBOW,
    R_FOOT_INDEX,
    R_HEEL,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    eval_cog_front,
    eval_cog_side,
    eval_force_sequence,
    eval_retract_speed_side,
    eval_wrist_angle,
)


def _empty_frame() -> np.ndarray:
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 3] = 1.0
    return lm


def _set_xy(lm: np.ndarray, idx: int, x: float, y: float) -> None:
    lm[idx, 0] = float(x)
    lm[idx, 1] = float(y)


def _assert_no_digits(s: str) -> None:
    if any(ch.isdigit() for ch in s):
        raise AssertionError(f"reason contains digits: {s!r}")


def _make_side_pose_frame(*, back_knee_straight: bool, front_knee_straight: bool, center_by_knee: bool) -> np.ndarray:
    lm = _empty_frame()

    _set_xy(lm, NOSE, 0.7, 0.2)
    _set_xy(lm, L_HIP, 0.5, 0.5)
    _set_xy(lm, R_HIP, 0.5, 0.5)

    _set_xy(lm, L_ANKLE, 0.62, 0.85)
    _set_xy(lm, R_ANKLE, 0.42, 0.85)
    _set_xy(lm, L_HEEL, 0.58, 0.86)
    _set_xy(lm, L_FOOT_INDEX, 0.66, 0.86)
    _set_xy(lm, R_HEEL, 0.38, 0.86)
    _set_xy(lm, R_FOOT_INDEX, 0.46, 0.86)

    if front_knee_straight:
        _set_xy(lm, L_HIP, 0.55, 0.55)
        _set_xy(lm, L_KNEE, 0.55, 0.70)
        _set_xy(lm, L_ANKLE, 0.55, 0.85)
    else:
        _set_xy(lm, L_HIP, 0.55, 0.55)
        _set_xy(lm, L_KNEE, 0.58, 0.70)
        _set_xy(lm, L_ANKLE, 0.55, 0.85)

    if back_knee_straight:
        _set_xy(lm, R_HIP, 0.45, 0.55)
        _set_xy(lm, R_KNEE, 0.45, 0.70)
        _set_xy(lm, R_ANKLE, 0.45, 0.85)
    else:
        _set_xy(lm, R_HIP, 0.45, 0.55)
        _set_xy(lm, R_KNEE, 0.42, 0.70)
        _set_xy(lm, R_ANKLE, 0.45, 0.85)

    if center_by_knee:
        _set_xy(lm, L_HEEL, 0.55, 0.86)
        _set_xy(lm, L_FOOT_INDEX, 0.65, 0.86)
        _set_xy(lm, L_KNEE, 0.60, 0.70)

    return lm


def _make_front_pose_frame(*, knee_straight: bool, ankle_acute: bool) -> np.ndarray:
    lm = _empty_frame()

    if knee_straight:
        _set_xy(lm, L_HIP, 0.5, 0.55)
        _set_xy(lm, L_KNEE, 0.5, 0.70)
        _set_xy(lm, L_ANKLE, 0.5, 0.85)
    else:
        _set_xy(lm, L_HIP, 0.5, 0.55)
        _set_xy(lm, L_KNEE, 0.53, 0.70)
        _set_xy(lm, L_ANKLE, 0.5, 0.85)

    if ankle_acute:
        _set_xy(lm, L_FOOT_INDEX, 0.62, 0.86)
    else:
        _set_xy(lm, L_FOOT_INDEX, 0.5, 0.98)

    return lm


def _make_retract_sequence(*, fps: float, slow: bool) -> np.ndarray:
    n = 20
    seq = np.stack([_empty_frame() for _ in range(n)], axis=0)
    for i in range(n):
        lm = seq[i]
        _set_xy(lm, NOSE, 0.7, 0.2)
        _set_xy(lm, L_HIP, 0.5, 0.55)
        _set_xy(lm, R_HIP, 0.5, 0.55)

        _set_xy(lm, L_SHOULDER, 0.45, 0.35)
        _set_xy(lm, L_ELBOW, 0.55, 0.35)
        _set_xy(lm, L_WRIST, 0.65, 0.35)

        _set_xy(lm, R_SHOULDER, 0.55, 0.35)
        _set_xy(lm, R_ELBOW, 0.55, 0.45)
        _set_xy(lm, R_WRIST, 0.55, 0.55)

    ext_end = 2
    if slow:
        ret_start = ext_end + int(round(0.25 * fps))
    else:
        ret_start = ext_end + int(round(0.10 * fps))

    for i in range(ret_start, n):
        lm = seq[i]
        _set_xy(lm, L_SHOULDER, 0.45, 0.35)
        _set_xy(lm, L_ELBOW, 0.52, 0.35)
        _set_xy(lm, L_WRIST, 0.46, 0.39)

    return seq


def main() -> None:
    fps = 30.0

    side_forward = np.stack([_make_side_pose_frame(back_knee_straight=True, front_knee_straight=False, center_by_knee=False) for _ in range(25)], axis=0)
    r = eval_cog_side(side_forward, fps=fps)
    assert r.status == "不合格" and "偏前" in r.reason

    side_backward = np.stack([_make_side_pose_frame(back_knee_straight=False, front_knee_straight=True, center_by_knee=False) for _ in range(25)], axis=0)
    r = eval_cog_side(side_backward, fps=fps)
    assert r.status == "不合格" and "偏后" in r.reason

    side_center = np.stack([_make_side_pose_frame(back_knee_straight=False, front_knee_straight=False, center_by_knee=True) for _ in range(25)], axis=0)
    r = eval_cog_side(side_center, fps=fps)
    assert r.status == "合格" and "居中" in r.reason

    front_forward = np.stack([_make_front_pose_frame(knee_straight=False, ankle_acute=True) for _ in range(25)], axis=0)
    r = eval_cog_front(front_forward, fps=fps, stance="left")
    assert r.status == "不合格" and "偏前" in r.reason

    retract_fast = eval_retract_speed_side(_make_retract_sequence(fps=fps, slow=False), fps=fps, min_punches=1)
    assert retract_fast.status == "合格"
    _assert_no_digits(retract_fast.reason)

    retract_slow = eval_retract_speed_side(_make_retract_sequence(fps=fps, slow=True), fps=fps, min_punches=1)
    assert retract_slow.status == "不合格"
    _assert_no_digits(retract_slow.reason)

    retract_insufficient = eval_retract_speed_side(_make_retract_sequence(fps=fps, slow=True), fps=fps, min_punches=4)
    assert retract_insufficient.status == "无法判定"
    _assert_no_digits(retract_insufficient.reason)

    wrist = eval_wrist_angle(_make_retract_sequence(fps=fps, slow=False), fps=fps, min_events=1)
    assert wrist.status in ("合格", "不合格", "无法判定")
    _assert_no_digits(wrist.reason)

    force = eval_force_sequence(np.zeros((0, 33, 4), dtype=np.float32), _make_retract_sequence(fps=fps, slow=False), fps=fps, stance="left")
    assert force.status in ("合格", "不合格", "无法判定")
    _assert_no_digits(force.reason)

    for s in (side_forward, side_backward, side_center, front_forward):
        assert s.shape == (25, 33, 4)

    print("selfcheck_tech_eval: OK")


if __name__ == "__main__":
    main()

