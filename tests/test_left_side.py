#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试左侧标准动作视频"""

from pathlib import Path
from analysis.tech_eval import evaluate_video_full

# 尝试找到左侧视频
possible_paths = [
    Path("标准动作视频--分解版/肖 武术散打标准动作视频汇总/直拳 左侧.mp4"),
    Path("标准动作视频--分解版/直拳/直拳 左侧.mp4"),
    Path("标准动作视频--分解版/直拳/直拳 左侧 .mp4"),
]

video_path = None
for p in possible_paths:
    if p.exists():
        video_path = p
        break

if not video_path:
    print("找不到左侧视频文件")
else:
    print(f"正在处理: {video_path}")
    result = evaluate_video_full(video_path, pose_variant="full", stance="left", view_hint="auto")
    
    force_seq = result.get("force_sequence", {})
    print(f"\n状态: {force_seq.get('status')}")
    print(f"原因: {force_seq.get('reason')}")
    
    detail = force_seq.get("detail", {})
    print(f"\n总出拳事件: {detail.get('events_total', 0)}")
    print(f"蹬地检测数: {detail.get('push_detected', 0)}")
    print(f"转腰检测数: {detail.get('twist_detected', 0)}")
    print(f"送肩检测数: {detail.get('drive_detected', 0)}")
