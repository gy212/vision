#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析标准动作视频的NPZ文件，提取发力顺序相关数据"""

import numpy as np
from pathlib import Path

from analysis.tech_eval import _detect_retract_events_side, _infer_front_leg_side

# 目录路径
NPZ_DIR = Path("outputs/标准动作视频--分解版_骨架_20260130_184655/直拳")
VIS_THR = 0.5

def analyze_npz(file_path: Path):
    """分析单个NPZ文件"""
    print(f"\n{'='*60}")
    print(f"分析文件: {file_path.name}")
    print('='*60)
    
    data = np.load(file_path, allow_pickle=True)
    
    landmarks = data['landmarks']  # [T, 33, 4]
    meta = data['meta'].item() if 'meta' in data else {}
    
    fps = meta.get('fps', 30.0)
    print(f"帧数: {landmarks.shape[0]}, FPS: {fps}")
    
    # BlazePose 33 关键点索引
    NOSE = 0
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HEEL, L_FOOT_INDEX, L_HIP, L_KNEE, L_ANKLE = 29, 31, 23, 25, 27
    R_HEEL, R_FOOT_INDEX, R_HIP, R_KNEE, R_ANKLE = 30, 32, 24, 26, 28
    
    def get_xy(lm, idx):
        return lm[idx, :2]

    def vis_ok(lm, idxs):
        return all(float(lm[i, 3]) >= VIS_THR for i in idxs)

    def frame_dir(lm):
        if float(lm[NOSE, 3]) < VIS_THR:
            return None
        if not vis_ok(lm, (L_HIP, R_HIP)):
            return None
        hip_x = float((lm[L_HIP, 0] + lm[R_HIP, 0]) * 0.5)
        dx = float(lm[NOSE, 0]) - hip_x
        if not np.isfinite(dx):
            return None
        return 1.0 if dx >= 0.0 else -1.0

    def foot_center_x(lm, side: str):
        if side.upper() == "L":
            heel, toe = L_HEEL, L_FOOT_INDEX
        else:
            heel, toe = R_HEEL, R_FOOT_INDEX
        if not vis_ok(lm, (heel, toe)):
            return None
        return float((lm[heel, 0] + lm[toe, 0]) * 0.5)

    def foot_angle_deg(lm, side: str):
        if side.upper() == "L":
            heel, toe = L_HEEL, L_FOOT_INDEX
        else:
            heel, toe = R_HEEL, R_FOOT_INDEX
        if not vis_ok(lm, (heel, toe)):
            return None
        a = get_xy(lm, heel)
        b = get_xy(lm, toe)
        return float(np.degrees(np.arctan2(float(b[1] - a[1]), float(b[0] - a[0]))))
    
    def calc_angle(a, b, c):
        """计算三点夹角（度）"""
        ba = a - b
        bc = c - b
        cos_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        return np.degrees(np.arccos(cos_ang))
    
    def calc_heel_lift(lm, is_left=True):
        """计算脚跟抬高程度"""
        if is_left:
            heel, toe = L_HEEL, L_FOOT_INDEX
        else:
            heel, toe = R_HEEL, R_FOOT_INDEX
        
        heel_xy = get_xy(lm, heel)
        toe_xy = get_xy(lm, toe)
        
        dy = toe_xy[1] - heel_xy[1]  # y向下为正
        dx = toe_xy[0] - heel_xy[0]
        
        if dy <= 0:
            return 0.0, 0.0
        
        pitch = np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-9))
        dy_ratio = dy / (abs(dx) + 1e-6)
        
        return pitch, dy_ratio
    
    def calc_knee_angle(lm, is_left=True):
        """计算膝盖角度"""
        if is_left:
            hip, knee, ankle = L_HIP, L_KNEE, L_ANKLE
        else:
            hip, knee, ankle = R_HIP, R_KNEE, R_ANKLE
        
        return calc_angle(get_xy(lm, hip), get_xy(lm, knee), get_xy(lm, ankle))
    
    is_front = ("正面" in file_path.name) or ("front" in file_path.name.lower())
    is_side = ("左侧" in file_path.name) or ("右侧" in file_path.name) or ("侧面" in file_path.name)

    # 分析后腿（假设左势，后腿是右脚）
    print("\n【右脚（后腿）蹬地分析】")
    print("-" * 60)
    
    pitch_values = []
    dy_ratio_values = []
    knee_angle_values = []
    
    for i, lm in enumerate(landmarks):
        pitch, dy_ratio = calc_heel_lift(lm, is_left=False)
        knee_ang = calc_knee_angle(lm, is_left=False)
        
        pitch_values.append(pitch)
        dy_ratio_values.append(dy_ratio)
        knee_angle_values.append(knee_ang)
    
    pitch_values = np.array(pitch_values)
    dy_ratio_values = np.array(dy_ratio_values)
    knee_angle_values = np.array(knee_angle_values)
    
    print(f"Pitch (脚底角度) 统计:")
    print(f"  最小: {pitch_values.min():.1f}°, 最大: {pitch_values.max():.1f}°")
    print(f"  平均: {pitch_values.mean():.1f}°, 中位数: {np.median(pitch_values):.1f}°")
    print(f"  P25: {np.percentile(pitch_values, 25):.1f}°, P75: {np.percentile(pitch_values, 75):.1f}°")
    
    print(f"\n脚跟抬高比例 (dy/|dx|) 统计:")
    print(f"  最小: {dy_ratio_values.min():.3f}, 最大: {dy_ratio_values.max():.3f}")
    print(f"  平均: {dy_ratio_values.mean():.3f}, 中位数: {np.median(dy_ratio_values):.3f}")
    print(f"  P25: {np.percentile(dy_ratio_values, 25):.3f}, P75: {np.percentile(dy_ratio_values, 75):.3f}")
    
    print(f"\n右腿膝盖角度 统计:")
    print(f"  最小: {knee_angle_values.min():.1f}°, 最大: {knee_angle_values.max():.1f}°")
    print(f"  平均: {knee_angle_values.mean():.1f}°, 中位数: {np.median(knee_angle_values):.1f}°")
    print(f"  P25: {np.percentile(knee_angle_values, 25):.1f}°, P75: {np.percentile(knee_angle_values, 75):.1f}°")
    
    # 检测可能的蹬地帧（高pitch + 高dy_ratio + 合适的膝盖角度）
    print("\n【可能的蹬地帧分析】")
    print("-" * 60)
    
    # 使用当前阈值
    pitch_threshold = 25.0
    dy_threshold = 0.03
    dy_ratio_threshold = 0.4
    knee_min, knee_max = 130.0, 175.0
    
    push_off_frames = []
    for i in range(len(landmarks)):
        if (pitch_values[i] >= pitch_threshold and 
            dy_ratio_values[i] >= dy_ratio_threshold and
            knee_min <= knee_angle_values[i] <= knee_max):
            push_off_frames.append(i)
    
    print(f"当前阈值检测到的蹬地帧数: {len(push_off_frames)}")
    if push_off_frames:
        print(f"  帧号: {push_off_frames[:20]}{'...' if len(push_off_frames) > 20 else ''}")
    
    # 建议阈值
    print("\n【基于数据的阈值建议】")
    print("-" * 60)
    
    # 取P80的pitch和dy_ratio作为合理阈值
    pitch_p80 = np.percentile(pitch_values[pitch_values > 10], 80) if np.any(pitch_values > 10) else 30
    dy_ratio_p80 = np.percentile(dy_ratio_values[dy_ratio_values > 0.1], 80) if np.any(dy_ratio_values > 0.1) else 0.4
    
    print(f"建议 Pitch 阈值: {pitch_p80:.1f}° (当前: {pitch_threshold}°)")
    print(f"建议 dy_ratio 阈值: {dy_ratio_p80:.3f} (当前: {dy_ratio_threshold})")
    print(f"建议膝盖角度范围: {np.percentile(knee_angle_values, 20):.1f}° - {np.percentile(knee_angle_values, 90):.1f}°")
    
    result = {
        'file': file_path.name,
        'view': 'front' if is_front else ('side' if is_side else 'unknown'),
        'pitch_stats': {
            'min': float(pitch_values.min()),
            'max': float(pitch_values.max()),
            'mean': float(pitch_values.mean()),
            'median': float(np.median(pitch_values)),
            'p80': float(pitch_p80)
        },
        'dy_ratio_stats': {
            'min': float(dy_ratio_values.min()),
            'max': float(dy_ratio_values.max()),
            'mean': float(dy_ratio_values.mean()),
            'median': float(np.median(dy_ratio_values)),
            'p80': float(dy_ratio_p80)
        },
        'knee_stats': {
            'min': float(knee_angle_values.min()),
            'max': float(knee_angle_values.max()),
            'mean': float(knee_angle_values.mean()),
            'p20': float(np.percentile(knee_angle_values, 20)),
            'p90': float(np.percentile(knee_angle_values, 90))
        },
        'push_off_frames': push_off_frames
    }

    # 侧面：上步位移统计（前脚上步、后脚跟进）
    if is_side:
        events = _detect_retract_events_side(
            landmarks,
            fps=float(fps),
            vis_thr=VIS_THR,
            ext_angle_thr=160.0,
            retract_angle_thr=45.0,
            ext_forward_dx=0.02,
            retract_sec_thr=0.2,
            max_window_sec=1.0,
        )
        pre = int(round(0.3 * float(fps)))
        post = int(round(0.4 * float(fps)))
        front_leg = _infer_front_leg_side(landmarks, vis_thr=VIS_THR, fallback="L")
        back_leg = "R" if front_leg == "L" else "L"

        def series_x(lms, side: str):
            xs = []
            for lm in lms:
                d = frame_dir(lm)
                if d is None:
                    xs.append(np.nan)
                    continue
                cx = foot_center_x(lm, side)
                if cx is None:
                    xs.append(np.nan)
                    continue
                xs.append(float(cx) * float(d))
            return np.asarray(xs, dtype=np.float32)

        def baseline(vals):
            if vals.size == 0:
                return None
            n0 = max(1, int(round(vals.size * 0.2)))
            base = vals[:n0]
            base = base[np.isfinite(base)]
            if base.size == 0:
                return None
            return float(np.median(base))

        front_steps = []
        back_steps = []
        for e in events:
            t_punch = int(e.get("start") or 0)
            s = max(0, t_punch - pre)
            lm_pre = landmarks[s : t_punch + 1]
            lm_post = landmarks[t_punch : min(int(landmarks.shape[0]), int(t_punch + post + 1))]
            if lm_pre.size == 0 or lm_post.size == 0:
                continue
            front_x = series_x(lm_pre, front_leg)
            back_x = series_x(lm_post, back_leg)
            base_f = baseline(front_x)
            base_b = baseline(back_x)
            if base_f is not None:
                disp_f = front_x - float(base_f)
                front_steps.append(float(np.nanmax(disp_f)))
            if base_b is not None:
                disp_b = back_x - float(base_b)
                back_steps.append(float(np.nanmax(disp_b)))

        result["step_stats"] = {
            "front_leg": str(front_leg),
            "back_leg": str(back_leg),
            "front_steps": front_steps,
            "back_steps": back_steps,
        }

    # 正面：脚部旋转方差统计
    if is_front:
        angles_l = []
        angles_r = []
        for lm in landmarks:
            a = foot_angle_deg(lm, "L")
            b = foot_angle_deg(lm, "R")
            if a is not None and np.isfinite(a):
                angles_l.append(float(a))
            if b is not None and np.isfinite(b):
                angles_r.append(float(b))

        def var_deg(arr):
            if len(arr) < 12:
                return None
            ang = np.deg2rad(np.asarray(arr, dtype=np.float32))
            ang = np.unwrap(ang)
            ang_deg = np.rad2deg(ang)
            return float(np.var(ang_deg))

        result["rotation_var_l"] = var_deg(angles_l)
        result["rotation_var_r"] = var_deg(angles_r)

    return result

def main():
    if not NPZ_DIR.exists():
        print(f"目录不存在: {NPZ_DIR}")
        return
    
    npz_files = list(NPZ_DIR.glob("*.npz"))
    print(f"找到 {len(npz_files)} 个NPZ文件")
    
    all_results = []
    for npz_file in sorted(npz_files):
        try:
            result = analyze_npz(npz_file)
            all_results.append(result)
        except Exception as e:
            print(f"分析 {npz_file.name} 时出错: {e}")
    
    # 汇总建议
    print("\n" + "="*60)
    print("【综合阈值建议】")
    print("="*60)
    
    if all_results:
        avg_pitch_p80 = np.mean([r['pitch_stats']['p80'] for r in all_results])
        avg_dy_p80 = np.mean([r['dy_ratio_stats']['p80'] for r in all_results])
        avg_knee_p20 = np.mean([r['knee_stats']['p20'] for r in all_results])
        avg_knee_p90 = np.mean([r['knee_stats']['p90'] for r in all_results])
        
        print(f"平均 Pitch P80: {avg_pitch_p80:.1f}°")
        print(f"平均 dy_ratio P80: {avg_dy_p80:.3f}")
        print(f"平均膝盖角度 P20-P90: {avg_knee_p20:.1f}° - {avg_knee_p90:.1f}°")
        
        print("\n【推荐阈值设置】")
        print(f"  pitch_range = (20, {min(65, avg_pitch_p80 + 10):.0f})")
        print(f"  dy_threshold = {max(0.02, avg_dy_p80 * 0.7):.3f}")
        print(f"  dy_ratio_threshold = {max(0.3, avg_dy_p80 * 0.7):.2f}")
        print(f"  knee_range = ({max(120, avg_knee_p20 - 10):.0f}, {min(180, avg_knee_p90 + 5):.0f})")

        # 上步阈值建议
        front_steps = [v for r in all_results for v in r.get("step_stats", {}).get("front_steps", []) if np.isfinite(v)]
        back_steps = [v for r in all_results for v in r.get("step_stats", {}).get("back_steps", []) if np.isfinite(v)]
        if front_steps and back_steps:
            front_p30 = float(np.percentile(front_steps, 30))
            back_p30 = float(np.percentile(back_steps, 30))
            print("\n【上步位移统计】")
            print(f"前脚位移 P30: {front_p30:.3f}")
            print(f"后脚位移 P30: {back_p30:.3f}")
            print("\n【推荐上步阈值】")
            print(f"  step_front_min = {max(0.01, front_p30 * 0.9):.3f}")
            print(f"  step_back_min = {max(0.01, back_p30 * 0.9):.3f}")

        # 正面旋转方差阈值建议
        rot_vars = []
        for r in all_results:
            if r.get("view") != "front":
                continue
            v1 = r.get("rotation_var_l")
            v2 = r.get("rotation_var_r")
            for v in (v1, v2):
                if v is not None and np.isfinite(v):
                    rot_vars.append(float(v))
        if rot_vars:
            rot_p95 = float(np.percentile(rot_vars, 95))
            print("\n【正面脚部旋转方差统计】")
            print(f"旋转方差 P95: {rot_p95:.1f} (deg^2)")
            print("\n【推荐旋转判死阈值】")
            print(f"  rotation_var_thr = {rot_p95:.1f}")

if __name__ == "__main__":
    main()
