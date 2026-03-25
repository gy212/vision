from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from core import action_compare as ac
from core.rule_scoring import extract_pose_raw


def _find_standard_video(std_dir: Path, *, kind: str) -> Path:
    """
    Find a front/side standard video in `std_dir` using filename heuristics.
    `kind` is "front" or "side".
    """
    if kind not in ("front", "side"):
        raise ValueError("kind must be 'front' or 'side'")

    vids = sorted([p for p in std_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi"}])
    if not vids:
        raise FileNotFoundError(f"No videos found in: {std_dir}")

    if kind == "front":
        keys = ("正面", "正", "front", "face", "前")
    else:
        keys = ("侧面", "侧", "side", "ce")

    for k in keys:
        for p in vids:
            if k.lower() in p.name.lower():
                return p

    raise FileNotFoundError(
        f"Cannot auto-detect {kind} standard video in {std_dir}. "
        f"Found: {[p.name for p in vids]}. Use --front/--side to specify explicitly."
    )


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description="Dual-template compare (front+side) + raw skeleton export.")
    ap.add_argument("--standard_dir", type=str, default=None, help="Folder containing standard front/side videos")
    ap.add_argument("--front", type=str, default=None, help="Explicit path to standard FRONT video")
    ap.add_argument("--side", type=str, default=None, help="Explicit path to standard SIDE video")
    ap.add_argument("--student_dir", type=str, required=True, help="Folder containing student videos (*.mp4)")
    ap.add_argument("--pose", default="full", choices=["lite", "full", "heavy"], help="Pose model variant")
    ap.add_argument("--out_dir", type=str, default=None, help="Output folder (default: ./outputs/dual_compare_<ts>)")
    ap.add_argument("--export_raw", action="store_true", help="Export raw pose landmarks (.npz) for std + student segments")
    ap.add_argument("--rules", action="store_true", help="Enable rule-based scoring")
    ap.add_argument("--action", default="both", choices=["stance", "punch", "both"], help="Rule action scope")
    ap.add_argument("--error-analysis", action="store_true", help="Export error analysis CSVs (rules + joints)")
    args = ap.parse_args()
    enable_error_analysis = bool(getattr(args, "error_analysis", False))

    std_dir = Path(args.standard_dir) if args.standard_dir else None
    student_dir = Path(args.student_dir)
    if not student_dir.exists():
        raise FileNotFoundError(f"student_dir not found: {student_dir}")

    if args.front:
        front_video = Path(args.front)
    else:
        if std_dir is None:
            raise ValueError("Need --standard_dir or --front")
        front_video = _find_standard_video(std_dir, kind="front")

    if args.side:
        side_video = Path(args.side)
    else:
        if std_dir is None:
            raise ValueError("Need --standard_dir or --side")
        side_video = _find_standard_video(std_dir, kind="side")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (Path(__file__).resolve().parent / "outputs" / f"dual_compare_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "templates").mkdir(parents=True, exist_ok=True)
    (out_dir / "skeleton").mkdir(parents=True, exist_ok=True)

    # 1) Create templates from the two standard videos.
    front_tpl = out_dir / "templates" / f"standard_front_{args.pose}.npz"
    side_tpl = out_dir / "templates" / f"standard_side_{args.pose}.npz"
    front_tpl = ac.create_template_from_video(front_video, pose_variant=args.pose, out_path=front_tpl)
    side_tpl = ac.create_template_from_video(side_video, pose_variant=args.pose, out_path=side_tpl)

    # 2) Batch compare all student videos.
    student_videos = sorted([p for p in student_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi"}])
    if not student_videos:
        raise FileNotFoundError(f"No student videos found in: {student_dir}")

    rows: list[dict[str, Any]] = []
    error_rules_rows: list[dict[str, Any]] = []
    error_joints_rows: list[dict[str, Any]] = []
    jsonl_path = out_dir / "compare_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for v in student_videos:
            res = ac.compare_video_to_dual_templates(
                front_tpl,
                side_tpl,
                v,
                workers=1,
                enable_rules=bool(args.rules),
                action_scope=str(args.action),
                enable_error_analysis=enable_error_analysis,
            )
            jf.write(json.dumps(_jsonable(res), ensure_ascii=False) + "\n")
            if enable_error_analysis and bool(args.rules):
                for view, violations in (
                    ("front", res.front_rule_violations),
                    ("side", res.side_rule_violations),
                ):
                    if not violations:
                        continue
                    for rv in violations:
                        error_rules_rows.append(
                            {
                                "video": str(v.name),
                                "view": str(view),
                                "rule_id": str(rv.rule_id),
                                "rule_name": str(rv.name),
                                "violation_ratio": float(rv.violation_ratio),
                                "penalty": int(rv.penalty),
                                "valid_frames": int(rv.valid_frames),
                                "total_frames": int(rv.total_frames),
                            }
                        )
            if enable_error_analysis:
                for view, joints in (
                    ("front", res.front_joint_errors),
                    ("side", res.side_joint_errors),
                ):
                    if not joints:
                        continue
                    for je in joints:
                        error_joints_rows.append(
                            {
                                "video": str(v.name),
                                "view": str(view),
                                "joint": str(je.joint),
                                "mean_dist": "" if je.mean_dist is None else float(je.mean_dist),
                                "p90_dist": "" if je.p90_dist is None else float(je.p90_dist),
                                "max_dist": "" if je.max_dist is None else float(je.max_dist),
                                "valid_frames": int(je.valid_frames),
                            }
                        )
            rows.append(
                {
                    "video": str(v),
                    "front_score": float(res.front_score),
                    "side_score": float(res.side_score),
                    "combined_percent": int(res.combined_percent),
                    "front_rule_score": "" if res.front_rule_score is None else int(res.front_rule_score),
                    "side_rule_score": "" if res.side_rule_score is None else int(res.side_rule_score),
                    "front_rule_deduction": "" if res.front_rule_deduction is None else int(res.front_rule_deduction),
                    "side_rule_deduction": "" if res.side_rule_deduction is None else int(res.side_rule_deduction),
                    "front_segment": "" if res.front_segment is None else f"{res.front_segment[0]}..{res.front_segment[1]}",
                    "side_segment": "" if res.side_segment is None else f"{res.side_segment[0]}..{res.side_segment[1]}",
                    "front_matches": int(len(res.front_matches)),
                    "side_matches": int(len(res.side_matches)),
                    "pose_variant": str(res.pose_variant),
                    "fps": float(res.fps),
                }
            )

    # Export standard raw (front/side) if requested.
    if args.export_raw:
        std_front_raw, meta_f = extract_pose_raw(front_video, pose_variant=args.pose)
        std_side_raw, meta_s = extract_pose_raw(side_video, pose_variant=args.pose)
        np.savez_compressed(out_dir / "skeleton" / f"standard_front_raw_{args.pose}.npz", landmarks=std_front_raw, meta=np.array(meta_f, dtype=object))
        np.savez_compressed(out_dir / "skeleton" / f"standard_side_raw_{args.pose}.npz", landmarks=std_side_raw, meta=np.array(meta_s, dtype=object))

        # Export student segments raw.
        # Re-read the jsonl for segments to avoid keeping all results in memory as dataclasses.
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            v = Path(d["video_path"])
            stem = v.stem

            fseg = d.get("front_segment")
            sseg = d.get("side_segment")
            if fseg:
                s, e = int(fseg[0]), int(fseg[1])
                arr, meta = extract_pose_raw(v, pose_variant=args.pose, start_frame=s, end_frame=e)
                meta["segment_kind"] = "front"
                np.savez_compressed(out_dir / "skeleton" / f"{stem}_front_raw_{args.pose}.npz", landmarks=arr, meta=np.array(meta, dtype=object))
            if sseg:
                s, e = int(sseg[0]), int(sseg[1])
                arr, meta = extract_pose_raw(v, pose_variant=args.pose, start_frame=s, end_frame=e)
                meta["segment_kind"] = "side"
                np.savez_compressed(out_dir / "skeleton" / f"{stem}_side_raw_{args.pose}.npz", landmarks=arr, meta=np.array(meta, dtype=object))

    # 3) CSV summary (UTF-8 BOM for Excel).
    csv_path = out_dir / "compare_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if enable_error_analysis and bool(args.rules):
        error_rules_path = out_dir / "error_rules.csv"
        with error_rules_path.open("w", encoding="utf-8-sig", newline="") as f:
            fieldnames = ["video", "view", "rule_id", "rule_name", "violation_ratio", "penalty", "valid_frames", "total_frames"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(error_rules_rows)

    if enable_error_analysis:
        error_joints_path = out_dir / "error_joints.csv"
        with error_joints_path.open("w", encoding="utf-8-sig", newline="") as f:
            fieldnames = ["video", "view", "joint", "mean_dist", "p90_dist", "max_dist", "valid_frames"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(error_joints_rows)

    print(f"Saved templates: {front_tpl} , {side_tpl}")
    print(f"Saved results:   {csv_path}")
    print(f"Saved jsonl:     {jsonl_path}")
    if args.export_raw:
        print(f"Saved skeleton:  {out_dir / 'skeleton'}")
    if enable_error_analysis and bool(args.rules):
        print(f"Saved error rules: {out_dir / 'error_rules.csv'}")
    if enable_error_analysis:
        print(f"Saved error joints: {out_dir / 'error_joints.csv'}")


if __name__ == "__main__":
    main()
