#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""详细分析2.mp4的问题"""

import numpy as np
from pathlib import Path
from analysis.tech_eval import evaluate_video_full, extract_pose_and_view_scores
import json

VIDEO_PATH = Path("学员样本/2.mp4")

def main():
    print(f"正在详细分析: {VIDEO_PATH}")
    print("=" * 70)
    
    # 提取数据
    landmarks, view_scores, meta = extract_pose_and_view_scores(
        VIDEO_PATH, pose_variant="full"
    )
    
    print(f"视频信息: {meta['frame_count']}帧, {meta['fps']}fps")
    print(f"视角分数范围: {np.nanmin(view_scores):.2f} - {np.nanmax(view_scores):.2f}")
    print(f"视角分数中位数: {np.nanmedian(view_scores):.2f}")
    
    # 运行完整评估
    result = evaluate_video_full(VIDEO_PATH, pose_variant="full", stance="left", view_hint="auto")
    
    # 详细分析发力顺序
    force_seq = result.get("force_sequence", {})
    detail = force_seq.get("detail", {})
    
    print("\n" + "=" * 70)
    print("【发力顺序详细分析】")
    print("=" * 70)
    print(f"状态: {force_seq.get('status')}")
    print(f"原因: {force_seq.get('reason')}")
    print(f"\n总事件: {detail.get('events_total')}")
    print(f"蹬地检测: {detail.get('push_detected')}/{detail.get('events_used')}")
    print(f"转腰检测: {detail.get('twist_detected')}/{detail.get('events_used')}")
    print(f"送肩检测: {detail.get('drive_detected')}/{detail.get('events_used')}")
    print(f"前腿: {detail.get('front_leg')}, 后腿: {detail.get('back_leg')}")
    
    # 分析每个事件
    events = detail.get("events", [])
    
    print("\n【逐事件分析】")
    print("-" * 70)
    
    # BlazePose 33 关键点索引
    L_HEEL, L_FOOT_INDEX, L_HIP, L_KNEE, L_ANKLE = 29, 31, 23, 25, 27
    R_HEEL, R_FOOT_INDEX, R_HIP, R_KNEE, R_ANKLE = 30, 32, 24, 26, 28
    L_SHOULDER, R_SHOULDER = 11, 12
    
    for i, e in enumerate(events):
        arm = e.get("arm", "?")
        t_push = e.get("t_push")
        t_twist = e.get("t_twist")
        t_drive = e.get("t_drive")
        t_punch = e.get("t_punch")
        ok = e.get("ok")
        dyn_score = e.get("push_dynamic_score")
        
        status = "✓ OK" if ok else ("✗ FAIL" if ok is False else "? N/A")
        print(f"\n事件{i+1} ({arm}): {status}")
        print(f"  t_push={t_push}, t_twist={t_twist}, t_drive={t_drive}, t_punch={t_punch}")
        
        if dyn_score:
            print(f"  动态检测分数: {dyn_score:.3f}")
        
        # 如果检测到了蹬地，分析该帧的后脚数据
        if t_push is not None and t_push < len(landmarks):
            lm = landmarks[t_push]
            
            # 判断后腿
            back_leg = "R" if detail.get("front_leg") == "L" else "L"
            
            if back_leg == "L":
                heel, toe, hip, knee, ankle = L_HEEL, L_FOOT_INDEX, L_HIP, L_KNEE, L_ANKLE
            else:
                heel, toe, hip, knee, ankle = R_HEEL, R_FOOT_INDEX, R_HIP, R_KNEE, R_ANKLE
            
            # 获取关键点坐标
            heel_xy = lm[heel, :2]
            toe_xy = lm[toe, :2]
            hip_xy = lm[hip, :2]
            knee_xy = lm[knee, :2]
            ankle_xy = lm[ankle, :2]
            
            # 计算脚跟抬高
            dy = toe_xy[1] - heel_xy[1]
            dx = toe_xy[0] - heel_xy[0]
            pitch = np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-9))
            dy_ratio = dy / (abs(dx) + 1e-6) if dx != 0 else 0
            
            # 计算膝盖角度
            ba = hip_xy - knee_xy
            bc = ankle_xy - knee_xy
            cos_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
            cos_ang = np.clip(cos_ang, -1.0, 1.0)
            knee_ang = np.degrees(np.arccos(cos_ang))
            
            print(f"  后腿({back_leg})检测帧数据:")
            print(f"    pitch={pitch:.1f}°, dy_ratio={dy_ratio:.3f}, knee_ang={knee_ang:.1f}°")
            print(f"    heel_xy=({heel_xy[0]:.3f}, {heel_xy[1]:.3f})")
            print(f"    toe_xy=({toe_xy[0]:.3f}, {toe_xy[1]:.3f})")
            
            # 检查关键点可见度
            vis_heel = lm[heel, 3]
            vis_toe = lm[toe, 3]
            vis_knee = lm[knee, 3]
            vis_ankle = lm[ankle, 3]
            print(f"    visibility: heel={vis_heel:.2f}, toe={vis_toe:.2f}, knee={vis_knee:.2f}, ankle={vis_ankle:.2f}")
            
            # 计算两脚间距（判断是否左右摇摆）
            l_heel = lm[L_HEEL, :2]
            r_heel = lm[R_HEEL, :2]
            foot_distance = np.linalg.norm(l_heel - r_heel)
            print(f"    两脚间距: {foot_distance:.4f}")
    
    # 保存完整结果
    output_file = Path("outputs/analyze_2mp4.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n\n完整结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
