from __future__ import annotations

import numpy as np


def pose_view_score(pose_landmarks) -> float | None:
    """
    Heuristic "frontness" score for a single frame.

    Higher => more likely facing the camera (front view).
    Lower  => more likely side view.

    This uses only a few stable BlazePose landmarks (shoulders/hips) and combines:
      - normalized shoulder width (relative to torso length)
      - left/right visibility balance (front tends to be more symmetric)
    """
    if pose_landmarks is None:
        return None

    lm = pose_landmarks

    # BlazePose indices
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24

    def xy(i: int) -> np.ndarray:
        return np.array([lm[i].x, lm[i].y], dtype=np.float32)

    def vis(i: int) -> float:
        return float(getattr(lm[i], "visibility", 1.0))

    ls, rs = xy(L_SHOULDER), xy(R_SHOULDER)
    lh, rh = xy(L_HIP), xy(R_HIP)

    if (not np.isfinite(ls).all()) or (not np.isfinite(rs).all()) or (not np.isfinite(lh).all()) or (not np.isfinite(rh).all()):
        return None

    shoulder_w = float(np.linalg.norm(ls - rs))
    torso = float(np.linalg.norm((0.5 * (ls + rs)) - (0.5 * (lh + rh))))
    if (not np.isfinite(shoulder_w)) or (not np.isfinite(torso)) or torso < 1e-6:
        return None

    width_ratio = shoulder_w / (torso + 1e-6)

    # Visibility balance helps when one side is occluded (typical in side view).
    v_ls, v_rs = vis(L_SHOULDER), vis(R_SHOULDER)
    v_lh, v_rh = vis(L_HIP), vis(R_HIP)

    def balance(a: float, b: float) -> float:
        denom = max(a, b, 1e-6)
        return 1.0 - (abs(a - b) / denom)

    vis_balance = 0.5 * (balance(v_ls, v_rs) + balance(v_lh, v_rh))

    s = float(width_ratio * vis_balance)
    if not np.isfinite(s):
        return None
    return s


def mirror_pose_features(features: np.ndarray) -> np.ndarray:
    """
    Mirror normalized pose features along the X axis and swap left/right joints.

    Input: (T,22,2) or (22,2) for BlazePose indices 11..32.
    """
    if features.ndim == 2:
        x = features.copy()
        x[:, 0] *= -1.0
        # Swap pairs (0,1), (2,3), ... in-place via a temp copy.
        y = x.copy()
        for a in range(0, 22, 2):
            y[a] = x[a + 1]
            y[a + 1] = x[a]
        return y

    if features.ndim == 3:
        x = features.copy()
        x[:, :, 0] *= -1.0
        y = x.copy()
        for a in range(0, 22, 2):
            y[:, a] = x[:, a + 1]
            y[:, a + 1] = x[:, a]
        return y

    raise ValueError(f"Unsupported features shape: {features.shape}")


def normalize_pose_xy_v1(pose_landmarks) -> np.ndarray | None:
    """
    Legacy normalization (v1) kept for backward compatibility with older templates.
    """
    if pose_landmarks is None:
        return None

    lm = pose_landmarks

    # BlazePose indices
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24

    def xy(i: int) -> np.ndarray:
        return np.array([lm[i].x, lm[i].y], dtype=np.float32)

    ls, rs = xy(L_SHOULDER), xy(R_SHOULDER)
    lh, rh = xy(L_HIP), xy(R_HIP)

    center = 0.5 * (lh + rh)
    if not np.isfinite(center).all():
        center = 0.5 * (ls + rs)

    scale = float(np.linalg.norm(ls - rs))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.linalg.norm(lh - rh))
    if not np.isfinite(scale) or scale < 1e-6:
        return None

    v = rs - ls
    ang = float(np.arctan2(v[1], v[0]))
    ca, sa = float(np.cos(-ang)), float(np.sin(-ang))
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)

    feats: list[np.ndarray] = []
    for i in range(11, 33):
        p = xy(i)
        p = (p - center) / scale
        p = R @ p
        feats.append(p)

    out = np.stack(feats, axis=0)
    if not np.isfinite(out).all():
        return None
    return out


def normalize_pose_xy(pose_landmarks) -> np.ndarray | None:
    """
    Convert pose landmarks into a normalized feature tensor.

    Output shape: (22, 2) for landmark indices 11..32 (face removed).
    Normalization:
      - translate by hip center (or shoulder center fallback)
      - scale by shoulder width (or hip width fallback)
      - rotate to make shoulders horizontal (when available)
    """
    if pose_landmarks is None:
        return None

    lm = pose_landmarks

    # BlazePose indices
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24

    def xy(i: int) -> np.ndarray:
        return np.array([lm[i].x, lm[i].y], dtype=np.float32)

    ls, rs = xy(L_SHOULDER), xy(R_SHOULDER)
    lh, rh = xy(L_HIP), xy(R_HIP)

    center = 0.5 * (lh + rh)
    if not np.isfinite(center).all():
        center = 0.5 * (ls + rs)

    # In normalized image coords, shoulder/hip width should not be extremely tiny.
    # A very small scale usually means a bad detection and causes numeric blow-ups.
    min_scale = 0.02
    scale = float(np.linalg.norm(ls - rs))
    if (not np.isfinite(scale)) or (scale < min_scale):
        scale = float(np.linalg.norm(lh - rh))
    if (not np.isfinite(scale)) or (scale < min_scale):
        return None

    # rotation (align shoulders horizontally)
    v = rs - ls
    ang = float(np.arctan2(v[1], v[0]))  # radians
    ca, sa = float(np.cos(-ang)), float(np.sin(-ang))
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)

    feats: list[np.ndarray] = []
    for i in range(11, 33):
        p = xy(i)
        p = (p - center) / scale
        p = R @ p
        # Clip extreme outliers; if we get too many, treat as invalid below.
        p = np.clip(p, -5.0, 5.0)
        feats.append(p)

    out = np.stack(feats, axis=0)
    if (not np.isfinite(out).all()) or float(np.max(np.abs(out))) > 5.0:
        return None
    return out


def normalize_pose_xy_v3(pose_landmarks) -> np.ndarray | None:
    """
    Normalization v3: more view-robust for side angles.

    Compared to v2, this prefers torso length as the scale (less dependent on yaw),
    and uses a hybrid rotation strategy:
      - front-ish: align shoulders to horizontal
      - side-ish:  align torso to vertical

    Output shape: (22, 2) for landmark indices 11..32.
    """
    if pose_landmarks is None:
        return None

    lm = pose_landmarks

    # BlazePose indices
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24

    def xy(i: int) -> np.ndarray:
        return np.array([lm[i].x, lm[i].y], dtype=np.float32)

    ls, rs = xy(L_SHOULDER), xy(R_SHOULDER)
    lh, rh = xy(L_HIP), xy(R_HIP)

    if (not np.isfinite(ls).all()) or (not np.isfinite(rs).all()) or (not np.isfinite(lh).all()) or (not np.isfinite(rh).all()):
        return None

    sh_c = 0.5 * (ls + rs)
    hip_c = 0.5 * (lh + rh)
    center = hip_c if np.isfinite(hip_c).all() else sh_c
    if not np.isfinite(center).all():
        return None

    shoulder_w = float(np.linalg.norm(ls - rs))
    hip_w = float(np.linalg.norm(lh - rh))
    torso_len = float(np.linalg.norm(sh_c - hip_c))

    # Prefer torso length as scale; fall back to widths if needed.
    min_scale = 0.02
    scale = torso_len if (np.isfinite(torso_len) and torso_len >= min_scale) else max(shoulder_w, hip_w)
    if (not np.isfinite(scale)) or (scale < min_scale):
        return None

    # Decide rotation axis by "front-ish" heuristic.
    width_ratio = float(shoulder_w / (torso_len + 1e-6)) if np.isfinite(torso_len) else 0.0
    use_shoulders = bool(width_ratio >= 0.35)
    if use_shoulders:
        v = rs - ls
        target = 0.0  # align to +X axis
    else:
        v = sh_c - hip_c
        target = -float(np.pi) / 2.0  # align upwards (-Y)

    v_norm = float(np.linalg.norm(v))
    if (not np.isfinite(v_norm)) or v_norm < 1e-6:
        # Fallback: no rotation (still scaled/centered).
        ca, sa = 1.0, 0.0
    else:
        ang = float(np.arctan2(float(v[1]), float(v[0])))
        rot = float(target - ang)
        ca, sa = float(np.cos(rot)), float(np.sin(rot))
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)

    feats: list[np.ndarray] = []
    for i in range(11, 33):
        p = xy(i)
        p = (p - center) / float(scale)
        p = R @ p
        p = np.clip(p, -5.0, 5.0)
        feats.append(p)

    out = np.stack(feats, axis=0)
    if (not np.isfinite(out).all()) or float(np.max(np.abs(out))) > 5.0:
        return None
    return out


def motion_energy(seq: np.ndarray) -> np.ndarray:
    # seq: (T, D) with finite values
    d = np.diff(seq, axis=0)
    return np.sqrt(np.mean(d * d, axis=1))


def find_active_range(energy: np.ndarray, pad: int = 10) -> tuple[int, int]:
    """
    Pick the most "active" contiguous segment based on motion energy.
    Returns (start_frame, end_frame) inclusive, in original frame indices.
    """
    if energy.size == 0:
        return 0, 0

    # Smooth (simple moving average)
    k = 9
    if energy.size >= k:
        kernel = np.ones(k, dtype=np.float32) / k
        e = np.convolve(energy, kernel, mode="same")
    else:
        e = energy

    thr = float(np.percentile(e, 70))
    active = e > thr
    if not active.any():
        return 0, int(energy.size)  # energy is T-1

    best_s = best_e = 0
    cur_s = None
    for i, on in enumerate(active.tolist()):
        if on and cur_s is None:
            cur_s = i
        if (not on) and cur_s is not None:
            cur_e = i - 1
            if (cur_e - cur_s) > (best_e - best_s):
                best_s, best_e = cur_s, cur_e
            cur_s = None
    if cur_s is not None:
        cur_e = int(active.size - 1)
        if (cur_e - cur_s) > (best_e - best_s):
            best_s, best_e = cur_s, cur_e

    # energy index i corresponds to transition frame i->i+1, so map back to frames
    start = max(0, best_s - pad)
    end = min(int(active.size), best_e + 1 + pad)  # end frame index
    return start, end


def subsequence_dtw(query: np.ndarray, seq: np.ndarray) -> tuple[float, int, int]:
    """
    Subsequence DTW: find best matching subsequence of `seq` for `query`.
    Returns (cost, start_index, end_index) in seq indices (inclusive).
    """
    q = query.astype(np.float32).reshape(query.shape[0], -1)
    s = seq.astype(np.float32).reshape(seq.shape[0], -1)

    n, m = q.shape[0], s.shape[0]
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    prev = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up, 2 left
    dp[0, :] = 0.0

    for i in range(1, n + 1):
        qi = q[i - 1]
        for j in range(1, m + 1):
            sj = s[j - 1]
            cost = float(np.linalg.norm(qi - sj))
            a = dp[i - 1, j - 1]
            b = dp[i - 1, j]
            c = dp[i, j - 1]
            if a <= b and a <= c:
                dp[i, j] = cost + a
                prev[i, j] = 0
            elif b <= c:
                dp[i, j] = cost + b
                prev[i, j] = 1
            else:
                dp[i, j] = cost + c
                prev[i, j] = 2

    end = int(np.argmin(dp[n, 1:]) + 1)  # dp-space
    best_cost = float(dp[n, end])

    i, j = n, end
    while i > 0:
        p = int(prev[i, j])
        if p == 0:
            i -= 1
            j -= 1
        elif p == 1:
            i -= 1
        else:
            j -= 1
        if j <= 0:
            break
    start = max(1, j)  # dp-space

    return best_cost, start - 1, end - 1


def subsequence_dtw_with_path(query: np.ndarray, seq: np.ndarray) -> tuple[float, int, int, list[tuple[int, int]]]:
    q = query.astype(np.float32).reshape(query.shape[0], -1)
    s = seq.astype(np.float32).reshape(seq.shape[0], -1)

    n, m = q.shape[0], s.shape[0]
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    prev = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up, 2 left
    dp[0, :] = 0.0

    for i in range(1, n + 1):
        qi = q[i - 1]
        for j in range(1, m + 1):
            sj = s[j - 1]
            cost = float(np.linalg.norm(qi - sj))
            a = dp[i - 1, j - 1]
            b = dp[i - 1, j]
            c = dp[i, j - 1]
            if a <= b and a <= c:
                dp[i, j] = cost + a
                prev[i, j] = 0
            elif b <= c:
                dp[i, j] = cost + b
                prev[i, j] = 1
            else:
                dp[i, j] = cost + c
                prev[i, j] = 2

    end = int(np.argmin(dp[n, 1:]) + 1)
    best_cost = float(dp[n, end])

    path_rev: list[tuple[int, int]] = []
    i, j = n, end
    while i > 0 and j > 0:
        path_rev.append((int(i - 1), int(j - 1)))
        p = int(prev[i, j])
        if p == 0:
            i -= 1
            j -= 1
        elif p == 1:
            i -= 1
        else:
            j -= 1

    start = max(1, int(j))
    path = list(reversed(path_rev))
    return best_cost, start - 1, end - 1, path
