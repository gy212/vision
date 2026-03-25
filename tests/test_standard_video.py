#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试标准动作视频的发力顺序检测"""

import json
from pathlib import Path
from analysis.tech_eval import evaluate_video_full, export_debug_video

# 标准动作视频路径
VIDEO_PATH = Path("标准动作视频--分解版/肖 武术散打标准动作视频汇总/直拳 右侧.mp4")
OUTPUT_DIR = Path("outputs/test_standard_force_seq")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    video_path = VIDEO_PATH
    if not video_path.exists():
        print(f"视频文件不存在: {video_path}")
        # 尝试其他可能的路径
        alt_paths = [
            Path("标准动作视频--分解版/直拳/直拳 右侧.mp4"),
            Path("标准动作视频--分解版/直拳/直拳 右侧 .mp4"),
            Path("标准动作视频--分解版/肖 武术散打标准动作视频汇总/直拳 右侧.mp4"),
        ]
        found = False
        for alt in alt_paths:
            if alt.exists():
                print(f"使用替代路径: {alt}")
                video_path = alt
                found = True
                break
        if not found:
            print("找不到视频文件，请检查路径")
            return
    
    print(f"正在处理标准视频: {video_path}")
    print("=" * 60)
    
    try:
        # 运行完整评估
        result = evaluate_video_full(video_path, pose_variant="full", stance="left", view_hint="auto")
        
        # 保存完整结果
        output_file = OUTPUT_DIR / "standard_result.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n完整结果已保存到: {output_file}")
        
        # 打印发力顺序详情
        print("\n" + "=" * 60)
        print("【发力顺序检测结果】")
        print("=" * 60)
        
        force_seq = result.get("force_sequence", {})
        print(f"状态: {force_seq.get('status')}")
        print(f"原因: {force_seq.get('reason')}")
        
        detail = force_seq.get("detail", {})
        print(f"\n总出拳事件: {detail.get('events_total', 0)}")
        print(f"有效评估事件: {detail.get('events_used', 0)}")
        print(f"合格事件: {detail.get('events_ok', 0)}")
        print(f"不合格事件: {detail.get('events_bad', 0)}")
        print(f"蹬地检测数: {detail.get('push_detected', 0)}")
        print(f"转腰检测数: {detail.get('twist_detected', 0)}")
        print(f"送肩检测数: {detail.get('drive_detected', 0)}")
        print(f"前腿: {detail.get('front_leg', '未知')}, 后腿: {detail.get('back_leg', '未知')}")
        
        # 打印每个事件的详细信息
        events = detail.get("events", [])
        if events:
            print(f"\n【事件明细】")
            for i, e in enumerate(events):
                arm = e.get("arm", "?")
                ok = e.get("ok")
                t_push = e.get("t_push")
                t_twist = e.get("t_twist")
                t_drive = e.get("t_drive")
                t_punch = e.get("t_punch")
                dyn_score = e.get("push_dynamic_score")
                
                status_str = "✓ OK" if ok else ("✗ FAIL" if ok is False else "? N/A")
                dyn_str = f" (动态:{dyn_score:.2f})" if dyn_score else ""
                
                print(f"  事件{i+1:2d} ({arm}): {status_str}")
                print(f"         push={t_push}, twist={t_twist}, drive={t_drive}, punch={t_punch}{dyn_str}")
        
        # 导出调试视频
        print("\n" + "=" * 60)
        print("正在导出调试视频...")
        debug_path = OUTPUT_DIR / "debug_video.mp4"
        export_debug_video(video_path, debug_path, pose_variant="full", stance="left", view_hint="auto")
        print(f"调试视频已保存到: {debug_path}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
