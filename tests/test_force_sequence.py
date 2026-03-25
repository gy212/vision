#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试改进后的发力顺序检测"""

import json
from pathlib import Path
from analysis.tech_eval import evaluate_video_full

# 测试样本目录
SAMPLE_DIR = Path("学员样本")
OUTPUT_DIR = Path("outputs/test_force_seq_improved")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 获取所有mp4文件
    video_files = sorted(SAMPLE_DIR.glob("*.mp4"))
    
    results = []
    for video_path in video_files:
        print(f"\n正在处理: {video_path.name}")
        try:
            # 运行完整评估（包含detail）
            result = evaluate_video_full(video_path, pose_variant="full", stance="left", view_hint="auto")
            
            # 提取发力顺序相关信息
            force_seq = result.get("force_sequence", {})
            
            print(f"  发力顺序结果: {force_seq.get('status')} - {force_seq.get('reason')}")
            
            detail = force_seq.get("detail", {})
            events = detail.get("events", [])
            
            print(f"  检测到的出拳事件: {detail.get('events_total', 0)}")
            print(f"  有效评估事件: {detail.get('events_used', 0)}")
            print(f"  合格事件: {detail.get('events_ok', 0)}, 不合格事件: {detail.get('events_bad', 0)}")
            
            # 打印每个事件的详细信息
            for i, e in enumerate(events):
                arm = e.get("arm", "?")
                ok = e.get("ok")
                t_push = e.get("t_push")
                t_twist = e.get("t_twist")
                t_drive = e.get("t_drive")
                t_punch = e.get("t_punch")
                dyn_score = e.get("push_dynamic_score")
                
                status_str = "OK" if ok else ("FAIL" if ok is False else "N/A")
                dyn_str = f" (dynamic:{dyn_score:.2f})" if dyn_score else ""
                
                print(f"    事件{i+1}({arm}): {status_str} - push={t_push}, twist={t_twist}, drive={t_drive}, punch={t_punch}{dyn_str}")
            
            results.append({
                "video": video_path.name,
                "result": result
            })
            
        except Exception as e:
            print(f"  错误: {e}")
            results.append({
                "video": video_path.name,
                "error": str(e)
            })
    
    # 保存完整结果
    output_file = OUTPUT_DIR / "test_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n完整结果已保存到: {output_file}")
    
    # 打印汇总
    print("\n=== 汇总 ===")
    for r in results:
        video = r["video"]
        if "error" in r:
            print(f"{video}: 错误 - {r['error']}")
        else:
            force_seq = r["result"].get("force_sequence", {})
            status = force_seq.get("status", "未知")
            reason = force_seq.get("reason", "")
            print(f"{video}: {status} - {reason}")

if __name__ == "__main__":
    main()
