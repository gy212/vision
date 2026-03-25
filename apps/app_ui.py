from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from tkinter import BooleanVar, DoubleVar, IntVar, Scrollbar, StringVar, Text, Tk, Toplevel, filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.action_compare import compare_video_to_template, create_template_from_video
from analysis.tech_eval import evaluate_video_assets, evaluate_video_detail, export_debug_video, to_jsonable
from core.vision_pipeline import MediaPipePipeline, PipelineConfig


ACTION_LABELS_ZH = {
    "V_SIGN": "✌（V 手势）",
    "HANDS_UP": "双手举起",
    "LEFT_HAND_UP": "左手举起",
    "RIGHT_HAND_UP": "右手举起",
    "SQUAT": "下蹲",
}


class CollapsibleSection:
    """A collapsible section with a toggle header."""

    def __init__(self, parent: ttk.Frame, title: str, expanded: bool = False) -> None:
        self.parent = parent
        self._expanded = BooleanVar(value=expanded)

        self.header = ttk.Frame(parent)
        self._toggle_btn = ttk.Button(
            self.header,
            text=self._get_toggle_text(),
            command=self._toggle,
            width=len(title) + 4,
        )
        self._toggle_btn.pack(side="left", anchor="w")

        self.content = ttk.Frame(parent)
        if expanded:
            self.content.pack(fill="x", pady=(4, 0))

    def _get_toggle_text(self) -> str:
        arrow = "▾" if self._expanded.get() else "▸"
        return f"{arrow} {self._toggle_btn.cget('text').lstrip('▸▾ ') if hasattr(self, '_toggle_btn') else ''}"

    def _toggle(self) -> None:
        self._expanded.set(not self._expanded.get())
        current_text = self._toggle_btn.cget("text")
        base_text = current_text.lstrip("▸▾ ")
        arrow = "▾" if self._expanded.get() else "▸"
        self._toggle_btn.configure(text=f"{arrow} {base_text}")

        if self._expanded.get():
            self.content.pack(fill="x", pady=(4, 0))
        else:
            self.content.pack_forget()

    def set_title(self, title: str) -> None:
        arrow = "▾" if self._expanded.get() else "▸"
        self._toggle_btn.configure(text=f"{arrow} {title}", width=len(title) + 4)

    def pack_header(self, **kwargs) -> None:
        self.header.pack(**kwargs)

    def is_expanded(self) -> bool:
        return self._expanded.get()


class CompareWindow:
    def __init__(self, parent: Tk) -> None:
        self._win = Toplevel(parent)
        self._win.title("动作比对（模板匹配）")
        self._win.geometry("680x780")
        self._win.minsize(600, 650)

        # Template mode: "existing" or "generate"
        self.template_mode_var = StringVar(value="existing")
        self.base_video_var = StringVar(value="")
        self.template_var = StringVar(value="")
        self.target_video_var = StringVar(value="")

        self.pose_var = StringVar(value="heavy")
        self.workers_var = IntVar(value=1)
        self.start_var = StringVar(value="")
        self.end_var = StringVar(value="")

        self.save_preview_var = BooleanVar(value=True)
        self.preview_out_var = StringVar(value="")

        self.status_var = StringVar(value="就绪")
        self.result_var = StringVar(value="")
        self.progress_text_var = StringVar(value="")

        # Store last comparison result for display
        self._last_score: float | None = None
        self._last_match_info: str = ""

        self._stop_evt = threading.Event()
        self._worker: threading.Thread | None = None

        self._build()
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

    def is_open(self) -> bool:
        return bool(self._win.winfo_exists())

    def focus(self) -> None:
        try:
            self._win.deiconify()
            self._win.lift()
            self._win.focus_force()
        except Exception:
            pass

    def _build(self) -> None:
        outer = ttk.Frame(self._win, padding=12)
        outer.pack(fill="both", expand=True)

        # ===== Step 1: Template Preparation =====
        step1 = ttk.Labelframe(outer, text="① 准备模板", padding=10)
        step1.pack(fill="x")

        # Mode selection
        mode_row = ttk.Frame(step1)
        mode_row.pack(fill="x")
        ttk.Radiobutton(
            mode_row, text="使用已有模板", variable=self.template_mode_var, value="existing",
            command=self._toggle_template_mode
        ).pack(side="left")
        ttk.Radiobutton(
            mode_row, text="从视频生成", variable=self.template_mode_var, value="generate",
            command=self._toggle_template_mode
        ).pack(side="left", padx=(16, 0))

        # Existing template frame
        self._existing_frame = ttk.Frame(step1)
        self._existing_frame.pack(fill="x", pady=(8, 0))
        ttk.Label(self._existing_frame, text="模板文件(.npz)：").grid(row=0, column=0, sticky="w")
        ttk.Entry(self._existing_frame, textvariable=self.template_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Button(self._existing_frame, text="选择…", command=self._browse_template).grid(row=0, column=2, padx=(8, 0))
        self._existing_frame.columnconfigure(1, weight=1)

        # Generate from video frame
        self._generate_frame = ttk.Frame(step1)
        gen_row1 = ttk.Frame(self._generate_frame)
        gen_row1.pack(fill="x")
        ttk.Label(gen_row1, text="基准视频：").grid(row=0, column=0, sticky="w")
        ttk.Entry(gen_row1, textvariable=self.base_video_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Button(gen_row1, text="选择…", command=self._browse_base).grid(row=0, column=2, padx=(8, 0))
        gen_row1.columnconfigure(1, weight=1)

        # Advanced options (collapsible)
        self._advanced_section = CollapsibleSection(self._generate_frame, "高级选项", expanded=False)
        self._advanced_section.set_title("高级选项")
        self._advanced_section.pack_header(fill="x", pady=(8, 0))

        adv = self._advanced_section.content
        adv_row1 = ttk.Frame(adv)
        adv_row1.pack(fill="x", pady=(4, 0))
        ttk.Label(adv_row1, text="Pose 模型：").pack(side="left")
        ttk.Combobox(adv_row1, textvariable=self.pose_var, values=["lite", "full", "heavy"],
                     state="readonly", width=8).pack(side="left", padx=(6, 14))
        ttk.Label(adv_row1, text="线程数：").pack(side="left")
        ttk.Spinbox(adv_row1, from_=1, to=16, textvariable=self.workers_var, width=6).pack(side="left", padx=(6, 0))

        adv_row2 = ttk.Frame(adv)
        adv_row2.pack(fill="x", pady=(6, 0))
        ttk.Label(adv_row2, text="起始帧：").pack(side="left")
        ttk.Entry(adv_row2, textvariable=self.start_var, width=8).pack(side="left", padx=(6, 14))
        ttk.Label(adv_row2, text="结束帧：").pack(side="left")
        ttk.Entry(adv_row2, textvariable=self.end_var, width=8).pack(side="left", padx=(6, 0))

        self._gen_btn = ttk.Button(self._generate_frame, text="生成模板", command=self._gen_template)
        self._gen_btn.pack(fill="x", pady=(10, 0))

        # ===== Step 2: Target Video =====
        step2 = ttk.Labelframe(outer, text="② 选择待比对视频", padding=10)
        step2.pack(fill="x", pady=(12, 0))

        tgt_row = ttk.Frame(step2)
        tgt_row.pack(fill="x")
        ttk.Label(tgt_row, text="目标视频：").grid(row=0, column=0, sticky="w")
        ttk.Entry(tgt_row, textvariable=self.target_video_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Button(tgt_row, text="选择…", command=self._browse_target).grid(row=0, column=2, padx=(8, 0))
        tgt_row.columnconfigure(1, weight=1)

        preview_row = ttk.Frame(step2)
        preview_row.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(
            preview_row, text="导出匹配片段预览", variable=self.save_preview_var, command=self._toggle_preview
        ).pack(side="left")
        self._preview_entry = ttk.Entry(preview_row, textvariable=self.preview_out_var, width=30)
        self._preview_entry.pack(side="left", padx=(8, 0), fill="x", expand=True)
        self._preview_btn = ttk.Button(preview_row, text="保存位置…", command=self._choose_preview_out)
        self._preview_btn.pack(side="left", padx=(8, 0))

        # ===== Step 3: Execute =====
        step3 = ttk.Labelframe(outer, text="③ 执行比对", padding=10)
        step3.pack(fill="x", pady=(12, 0))

        btn_row = ttk.Frame(step3)
        btn_row.pack(fill="x")
        self.start_btn = ttk.Button(btn_row, text="═══ 开始比对 ═══", command=self._start_compare)
        self.start_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(btn_row, text="停止", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))

        # Progress bar (hidden initially)
        self._progress_frame = ttk.Frame(step3)
        self._progress_frame.pack(fill="x", pady=(8, 0))
        self.progress_bar = ttk.Progressbar(self._progress_frame, orient="horizontal", mode="determinate", maximum=100.0)
        self.progress_bar.pack(fill="x")
        ttk.Label(self._progress_frame, textvariable=self.progress_text_var).pack(anchor="w", pady=(4, 0))
        self._progress_frame.pack_forget()  # Hide initially

        # ===== Results =====
        result_frame = ttk.Labelframe(outer, text="结果", padding=10)
        result_frame.pack(fill="both", expand=True, pady=(12, 0))

        # Status line
        ttk.Label(result_frame, textvariable=self.status_var).pack(anchor="w")

        # Large score display
        self._score_frame = ttk.Frame(result_frame)
        self._score_frame.pack(fill="x", pady=(8, 0))

        self._score_label = ttk.Label(
            self._score_frame, text="—", font=("Helvetica", 36, "bold"), anchor="center"
        )
        self._score_label.pack()
        self._score_hint = ttk.Label(self._score_frame, text="相似度", anchor="center")
        self._score_hint.pack()

        # Match info
        self._match_label = ttk.Label(result_frame, textvariable=self.result_var, wraplength=600)
        self._match_label.pack(anchor="w", pady=(8, 0))

        # Collapsible JSON section
        self._json_section = CollapsibleSection(result_frame, "查看详细数据", expanded=False)
        self._json_section.set_title("查看详细数据")
        self._json_section.pack_header(fill="x", pady=(10, 0))

        json_content = self._json_section.content
        raw_box = ttk.Frame(json_content)
        raw_box.pack(fill="both", expand=True, pady=(4, 0))
        self.raw_text = Text(raw_box, height=10, wrap="none")
        self.raw_text.pack(side="left", fill="both", expand=True)
        sb = Scrollbar(raw_box, command=self.raw_text.yview)
        sb.pack(side="right", fill="y")
        self.raw_text.configure(yscrollcommand=sb.set)
        ttk.Button(json_content, text="复制数据", command=self._copy_raw).pack(anchor="e", pady=(6, 0))

        # Initialize display state
        self._toggle_template_mode()
        self._toggle_preview()
        self._reset_score_display()

    def _toggle_preview(self) -> None:
        enabled = bool(self.save_preview_var.get())
        self._preview_entry.configure(state="normal" if enabled else "disabled")
        self._preview_btn.configure(state="normal" if enabled else "disabled")
        if not enabled:
            self.preview_out_var.set("")

    def _toggle_template_mode(self) -> None:
        """Switch between existing template and generate-from-video modes."""
        mode = self.template_mode_var.get()
        if mode == "existing":
            self._generate_frame.pack_forget()
            self._existing_frame.pack(fill="x", pady=(8, 0))
        else:
            self._existing_frame.pack_forget()
            self._generate_frame.pack(fill="x", pady=(8, 0))

    def _reset_score_display(self) -> None:
        """Reset score display to initial state."""
        self._last_score = None
        self._last_match_info = ""
        self._score_label.configure(text="—", foreground="")
        self.result_var.set("")

    def _update_score_display(self, score: float, match_info: str = "") -> None:
        """Update the large score display with color coding."""
        self._last_score = score
        self._last_match_info = match_info

        # Display as percentage
        pct = int(score * 100)
        self._score_label.configure(text=f"{pct}%")

        # Color coding based on score
        if score >= 0.8:
            color = "#2e7d32"  # Green - very similar
        elif score >= 0.5:
            color = "#f9a825"  # Yellow/amber - partial
        else:
            color = "#c62828"  # Red - different

        self._score_label.configure(foreground=color)
        self.result_var.set(match_info)

    def _show_progress(self, show: bool = True) -> None:
        """Show or hide the progress frame."""
        if show:
            self._progress_frame.pack(fill="x", pady=(8, 0))
        else:
            self._progress_frame.pack_forget()

    def _browse_base(self) -> None:
        p = filedialog.askopenfilename(
            title="选择基准视频",
            filetypes=[("视频文件", "*.mp4;*.avi;*.mov;*.mkv"), ("所有文件", "*.*")],
            parent=self._win,
        )
        self._win.lift()
        self._win.focus_force()
        if p:
            self.base_video_var.set(p)

    def _browse_target(self) -> None:
        p = filedialog.askopenfilename(
            title="选择目标视频",
            filetypes=[("视频文件", "*.mp4;*.avi;*.mov;*.mkv"), ("所有文件", "*.*")],
            parent=self._win,
        )
        self._win.lift()
        self._win.focus_force()
        if p:
            self.target_video_var.set(p)

    def _browse_template(self) -> None:
        p = filedialog.askopenfilename(
            title="选择模板文件",
            filetypes=[("模板文件", "*.npz"), ("所有文件", "*.*")],
            parent=self._win,
        )
        self._win.lift()
        self._win.focus_force()
        if p:
            self.template_var.set(p)

    def _choose_preview_out(self) -> None:
        p = filedialog.asksaveasfilename(
            title="保存匹配预览视频",
            defaultextension=".mp4",
            filetypes=[("MP4 视频", "*.mp4"), ("AVI 视频", "*.avi"), ("所有文件", "*.*")],
            parent=self._win,
        )
        self._win.lift()
        self._win.focus_force()
        if p:
            self.preview_out_var.set(p)

    def _parse_int_or_none(self, s: str) -> int | None:
        s = (s or "").strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            raise ValueError(f"请输入整数帧号：{s!r}")

    def _progress(self, stage: str, done: int, total: int) -> None:
        def _set() -> None:
            self._show_progress(True)
            if total > 0:
                self.progress_bar.configure(mode="determinate", maximum=float(total))
                self.progress_bar["value"] = float(done)
                pct = (done / total) * 100.0
                self.progress_text_var.set(f"{stage}：{done}/{total}（{pct:.1f}%）")
            else:
                self.progress_bar.configure(mode="determinate", maximum=100.0)
                self.progress_bar["value"] = 0.0
                self.progress_text_var.set(f"{stage}…")

        self._win.after(0, _set)

    def _set_status(self, text: str) -> None:
        self._win.after(0, lambda: self.status_var.set(text))

    def _set_result(self, text: str) -> None:
        self._win.after(0, lambda: self.result_var.set(text))

    def _set_raw(self, text: str) -> None:
        def _set() -> None:
            self.raw_text.delete("1.0", "end")
            self.raw_text.insert("1.0", text)

        self._win.after(0, _set)

    def _copy_raw(self) -> None:
        try:
            s = self.raw_text.get("1.0", "end").strip()
            self._win.clipboard_clear()
            self._win.clipboard_append(s)
            self._set_status("已复制原始数据到剪贴板")
        except Exception as e:
            messagebox.showerror("复制失败", str(e), parent=self._win)

    def _set_buttons(self, running: bool) -> None:
        def _set() -> None:
            self.start_btn.configure(state="disabled" if running else "normal")
            self.stop_btn.configure(state="normal" if running else "disabled")
            if not running:
                self._show_progress(False)

        self._win.after(0, _set)

    def _gen_template(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        base = self.base_video_var.get().strip()
        if not base:
            messagebox.showerror("配置错误", "请先选择基准视频。", parent=self._win)
            return

        try:
            start = self._parse_int_or_none(self.start_var.get())
            end = self._parse_int_or_none(self.end_var.get())
        except Exception as e:
            messagebox.showerror("配置错误", str(e), parent=self._win)
            return

        self._stop_evt.clear()
        self._set_buttons(True)
        self._set_status("正在生成模板…")
        self._reset_score_display()
        self._set_raw("")
        self._progress("准备中", 0, 0)

        def _run() -> None:
            try:
                tpl_path = create_template_from_video(
                    base,
                    pose_variant=self.pose_var.get(),
                    start=start,
                    end=end,
                    workers=int(self.workers_var.get() or 1),
                    preview=False,
                    progress_cb=self._progress,
                    stop_evt=self._stop_evt,
                )
                self._win.after(0, lambda: self.template_var.set(str(tpl_path)))
                self._set_status("模板生成完成")
                self._set_result(f"模板：{tpl_path}")
                payload = {"template_path": str(tpl_path)}
                self._set_raw(json.dumps(payload, ensure_ascii=False, indent=2))
            except Exception as e:
                self._set_status("模板生成失败")
                self._set_result(str(e))
                self._set_raw("")
            finally:
                self._set_buttons(False)

        self._worker = threading.Thread(target=_run, daemon=True)
        self._worker.start()

    def _start_compare(self) -> None:
        if self._worker and self._worker.is_alive():
            return

        tpl = self.template_var.get().strip()
        vid = self.target_video_var.get().strip()
        if not tpl:
            messagebox.showerror("配置错误", "请先选择模板文件（.npz），或从基准视频生成模板。", parent=self._win)
            return
        if not vid:
            messagebox.showerror("配置错误", "请先选择目标视频。", parent=self._win)
            return

        preview_out = None
        if self.save_preview_var.get():
            preview_out = self.preview_out_var.get().strip()
            if not preview_out:
                preview_out = str(Path(tpl).with_suffix(".match.preview.avi"))
                self.preview_out_var.set(preview_out)

        self._stop_evt.clear()
        self._set_buttons(True)
        self._set_status("正在比对…")
        self._reset_score_display()
        self._set_raw("")
        self._progress("准备中", 0, 0)

        def _run() -> None:
            try:
                res = compare_video_to_template(
                    tpl,
                    vid,
                    pose_variant=self.pose_var.get(),
                    workers=int(self.workers_var.get() or 1),
                    preview_out=preview_out,
                    progress_cb=self._progress,
                    stop_evt=self._stop_evt,
                )
                # Show raw backend result + selected template meta for transparency/debugging.
                tpl_meta = {}
                try:
                    d = np.load(tpl, allow_pickle=True)
                    tpl_meta = d["meta"].item()
                except Exception:
                    tpl_meta = {}
                payload = {
                    "score": res.score,
                    "avg_cost": res.avg_cost,
                    "cost": res.cost,
                    "start_frame": res.start_frame,
                    "end_frame": res.end_frame,
                    "fps": res.fps,
                    "pose_variant": res.pose_variant,
                    "workers_used": res.workers_used,
                    "template_path": str(res.template_path),
                    "video_path": str(res.video_path),
                    "preview_path": str(res.preview_path) if res.preview_path else None,
                    "template_meta": tpl_meta,
                }
                self._set_raw(json.dumps(payload, ensure_ascii=False, indent=2))

                # Build match info message
                match_info = (
                    f"匹配片段：帧 {res.start_frame}..{res.end_frame}  "
                    f"时间 {res.start_frame/res.fps:.2f}s ~ {res.end_frame/res.fps:.2f}s"
                )
                if res.workers_used and res.workers_used != int(self.workers_var.get() or 1):
                    match_info += f"\n说明：模板为 VIDEO 模式，已自动使用单线程以保证准确。"
                if res.preview_path:
                    self._win.after(0, lambda: self.preview_out_var.set(str(res.preview_path)))
                    match_info += f"\n预览导出：{res.preview_path}"

                # Update the large score display
                def _update_ui() -> None:
                    self._update_score_display(res.score, match_info)

                self._win.after(0, _update_ui)
                self._set_status("比对完成")
            except Exception as e:
                self._set_status("比对失败")
                self._set_result(str(e))
                self._set_raw("")
            finally:
                self._set_buttons(False)

        self._worker = threading.Thread(target=_run, daemon=True)
        self._worker.start()

    def _stop(self) -> None:
        self._stop_evt.set()
        self._set_status("正在停止…")

    def _on_close(self) -> None:
        self._stop_evt.set()
        self._win.destroy()


class TechEvalWindow:
    def __init__(self, parent: Tk) -> None:
        self._win = Toplevel(parent)
        self._win.title("直拳技术指标评估")
        self._win.geometry("680x780")
        self._win.minsize(620, 680)

        self.video_dir_var = StringVar(value="")
        self.out_dir_var = StringVar(value="")
        self.pose_var = StringVar(value="full")
        self.stance_var = StringVar(value="left")
        self.view_var = StringVar(value="auto")
        self.full_var = BooleanVar(value=False)
        self.debug_video_var = BooleanVar(value=False)

        self.status_var = StringVar(value="就绪")
        self.result_var = StringVar(value="")
        self.progress_text_var = StringVar(value="")
        self.detail_text: Text | None = None

        self._stop_evt = threading.Event()
        self._worker: threading.Thread | None = None

        self._build()
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

    def is_open(self) -> bool:
        return bool(self._win.winfo_exists())

    def focus(self) -> None:
        try:
            self._win.deiconify()
            self._win.lift()
            self._win.focus_force()
        except Exception:
            pass

    def _build(self) -> None:
        outer = ttk.Frame(self._win, padding=12)
        outer.pack(fill="both", expand=True)

        step1 = ttk.Labelframe(outer, text="① 选择视频目录", padding=10)
        step1.pack(fill="x")
        row = ttk.Frame(step1)
        row.pack(fill="x")
        ttk.Label(row, text="视频目录：").grid(row=0, column=0, sticky="w")
        ttk.Entry(row, textvariable=self.video_dir_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Button(row, text="选择…", command=self._browse_dir).grid(row=0, column=2, padx=(8, 0))
        row.columnconfigure(1, weight=1)

        step2 = ttk.Labelframe(outer, text="② 选项", padding=10)
        step2.pack(fill="x", pady=(12, 0))
        opt_row = ttk.Frame(step2)
        opt_row.pack(fill="x")
        ttk.Label(opt_row, text="Pose：").pack(side="left")
        ttk.Combobox(opt_row, textvariable=self.pose_var, values=["lite", "full", "heavy"], state="readonly", width=8).pack(
            side="left", padx=(6, 14)
        )
        ttk.Label(opt_row, text="站姿：").pack(side="left")
        ttk.Combobox(opt_row, textvariable=self.stance_var, values=["left", "right"], state="readonly", width=8).pack(
            side="left", padx=(6, 14)
        )
        ttk.Label(opt_row, text="视角：").pack(side="left")
        ttk.Combobox(opt_row, textvariable=self.view_var, values=["auto", "front", "side"], state="readonly", width=8).pack(
            side="left", padx=(6, 0)
        )

        flags = ttk.Frame(step2)
        flags.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(flags, text="输出完整 JSONL（含明细）", variable=self.full_var).pack(side="left")
        ttk.Checkbutton(flags, text="导出调试视频", variable=self.debug_video_var).pack(side="left", padx=(16, 0))

        step3 = ttk.Labelframe(outer, text="③ 输出位置", padding=10)
        step3.pack(fill="x", pady=(12, 0))
        out_row = ttk.Frame(step3)
        out_row.pack(fill="x")
        ttk.Label(out_row, text="输出目录（可选）：").grid(row=0, column=0, sticky="w")
        ttk.Entry(out_row, textvariable=self.out_dir_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Button(out_row, text="选择…", command=self._browse_out_dir).grid(row=0, column=2, padx=(8, 0))
        out_row.columnconfigure(1, weight=1)

        step4 = ttk.Labelframe(outer, text="④ 执行", padding=10)
        step4.pack(fill="x", pady=(12, 0))
        btn_row = ttk.Frame(step4)
        btn_row.pack(fill="x")
        self.start_btn = ttk.Button(btn_row, text="═══ 开始评估 ═══", command=self._start)
        self.start_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(btn_row, text="停止", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))

        self._progress_frame = ttk.Frame(step4)
        self._progress_frame.pack(fill="x", pady=(8, 0))
        self.progress_bar = ttk.Progressbar(self._progress_frame, orient="horizontal", mode="determinate", maximum=100.0)
        self.progress_bar.pack(fill="x")
        ttk.Label(self._progress_frame, textvariable=self.progress_text_var).pack(anchor="w", pady=(4, 0))
        self._progress_frame.pack_forget()

        result_frame = ttk.Labelframe(outer, text="结果", padding=10)
        result_frame.pack(fill="both", expand=True, pady=(12, 0))
        ttk.Label(result_frame, textvariable=self.status_var).pack(anchor="w")
        ttk.Label(result_frame, textvariable=self.result_var, wraplength=620).pack(anchor="w", pady=(6, 0))

        detail_frame = ttk.Labelframe(result_frame, text="指标详情（当前视频）", padding=8)
        detail_frame.pack(fill="both", expand=False, pady=(8, 0))
        detail_box = ttk.Frame(detail_frame)
        detail_box.pack(fill="both", expand=True)
        self.detail_text = Text(detail_box, height=9, wrap="none")
        self.detail_text.pack(side="left", fill="both", expand=True)
        detail_sb = Scrollbar(detail_box, command=self.detail_text.yview)
        detail_sb.pack(side="right", fill="y")
        self.detail_text.configure(yscrollcommand=detail_sb.set, state="disabled")

        self._json_section = CollapsibleSection(result_frame, "查看详细数据", expanded=False)
        self._json_section.set_title("查看详细数据")
        self._json_section.pack_header(fill="x", pady=(10, 0))

        json_content = self._json_section.content
        raw_box = ttk.Frame(json_content)
        raw_box.pack(fill="both", expand=True, pady=(4, 0))
        self.raw_text = Text(raw_box, height=10, wrap="none")
        self.raw_text.pack(side="left", fill="both", expand=True)
        sb = Scrollbar(raw_box, command=self.raw_text.yview)
        sb.pack(side="right", fill="y")
        self.raw_text.configure(yscrollcommand=sb.set)
        ttk.Button(json_content, text="复制数据", command=self._copy_raw).pack(anchor="e", pady=(6, 0))

    def _show_progress(self, show: bool = True) -> None:
        if show:
            self._progress_frame.pack(fill="x", pady=(8, 0))
        else:
            self._progress_frame.pack_forget()

    def _progress(self, stage: str, done: int, total: int) -> None:
        def _set() -> None:
            self._show_progress(True)
            if total > 0:
                self.progress_bar.configure(mode="determinate", maximum=float(total))
                self.progress_bar["value"] = float(done)
                pct = (done / total) * 100.0
                self.progress_text_var.set(f"{stage}：{done}/{total}（{pct:.1f}%）")
            else:
                self.progress_bar.configure(mode="determinate", maximum=100.0)
                self.progress_bar["value"] = 0.0
                self.progress_text_var.set(f"{stage}…")

        self._win.after(0, _set)

    def _set_status(self, text: str) -> None:
        self._win.after(0, lambda: self.status_var.set(text))

    def _set_result(self, text: str) -> None:
        self._win.after(0, lambda: self.result_var.set(text))

    def _set_detail(self, text: str) -> None:
        if self.detail_text is None:
            return

        def _set() -> None:
            if self.detail_text is None:
                return
            self.detail_text.configure(state="normal")
            self.detail_text.delete("1.0", "end")
            self.detail_text.insert("1.0", text)
            self.detail_text.configure(state="disabled")

        self._win.after(0, _set)

    def _set_raw(self, text: str) -> None:
        def _set() -> None:
            self.raw_text.delete("1.0", "end")
            self.raw_text.insert("1.0", text)

        self._win.after(0, _set)

    def _copy_raw(self) -> None:
        try:
            s = self.raw_text.get("1.0", "end").strip()
            self._win.clipboard_clear()
            self._win.clipboard_append(s)
            self._set_status("已复制原始数据到剪贴板")
        except Exception as e:
            messagebox.showerror("复制失败", str(e), parent=self._win)

    def _set_buttons(self, running: bool) -> None:
        def _set() -> None:
            self.start_btn.configure(state="disabled" if running else "normal")
            self.stop_btn.configure(state="normal" if running else "disabled")
            if not running:
                self._show_progress(False)

        self._win.after(0, _set)

    def _browse_dir(self) -> None:
        p = filedialog.askdirectory(title="选择视频目录", parent=self._win)
        self._win.lift()
        self._win.focus_force()
        if p:
            self.video_dir_var.set(p)

    def _browse_out_dir(self) -> None:
        p = filedialog.askdirectory(title="选择输出目录", parent=self._win)
        self._win.lift()
        self._win.focus_force()
        if p:
            self.out_dir_var.set(p)

    def _iter_videos(self, root: Path) -> list[Path]:
        exts = {".mp4", ".mov", ".avi", ".mkv"}
        out: list[Path] = []
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                out.append(p)
        return sorted(out)

    def _start(self) -> None:
        if self._worker and self._worker.is_alive():
            return

        video_dir = Path(self.video_dir_var.get().strip())
        if not video_dir.exists():
            messagebox.showerror("配置错误", "请先选择一个存在的视频目录。", parent=self._win)
            return

        videos = self._iter_videos(video_dir)
        if not videos:
            messagebox.showerror("配置错误", "目录中未找到视频文件（mp4/mov/avi/mkv）。", parent=self._win)
            return

        out_dir_raw = self.out_dir_var.get().strip()
        if out_dir_raw:
            out_dir = Path(out_dir_raw)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path(__file__).resolve().parent / "outputs" / f"tech_eval_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "tech_report.csv"
        full_path = out_dir / "tech_report_full.jsonl"
        debug_dir = out_dir / "debug_videos"
        if self.debug_video_var.get():
            debug_dir.mkdir(parents=True, exist_ok=True)

        self._stop_evt.clear()
        self._set_buttons(True)
        self._set_status("正在评估…")
        self._set_result("")
        self._set_raw("")
        self._progress("准备中", 0, len(videos))

        pose_variant = self.pose_var.get().strip() or "full"
        stance = self.stance_var.get().strip() or "left"
        view_hint = self.view_var.get().strip() or "auto"
        want_full = bool(self.full_var.get())
        want_debug = bool(self.debug_video_var.get())

        def _run() -> None:
            rows: list[dict[str, str]] = []
            full_f = full_path.open("w", encoding="utf-8") if want_full else None
            done = 0

            def _cause(ind: object) -> str:
                detail = getattr(ind, "detail", None)
                if not isinstance(detail, dict):
                    return ""
                return str(detail.get("primary_cause") or "")

            def _failed_stage(ind: object) -> str | None:
                detail = getattr(ind, "detail", None)
                if not isinstance(detail, dict):
                    return None
                val = detail.get("failed_stage")
                return None if val in (None, "") else str(val)

            def _fmt_indicator(label: str, ind: object | None) -> str:
                if ind is None:
                    return f"{label}: 未评估"
                status = str(getattr(ind, "status", ""))
                reason = str(getattr(ind, "reason", ""))
                cause = _cause(ind)
                failed = _failed_stage(ind)
                parts = [f"{label}: {status}", reason]
                if cause:
                    parts.append(f"原因类型: {cause}")
                if failed:
                    parts.append(f"失败环节: {failed}")
                return " | ".join(p for p in parts if p)

            try:
                for v in videos:
                    if self._stop_evt.is_set():
                        break

                    if want_full or want_debug:
                        res_detail, lm, vs, meta = evaluate_video_assets(v, pose_variant=pose_variant, stance=stance, view_hint=view_hint)
                        res = res_detail
                    else:
                        res = evaluate_video_detail(v, pose_variant=pose_variant, stance=stance, view_hint=view_hint)
                        lm = vs = meta = None

                    row = {
                        "video": str(v.name),
                        "view_mode": str(res.view_mode),
                        "重心(侧面优先)": str(res.cog_final.status),
                        "重心说明": str(res.cog_final.reason),
                        "重心原因类型": _cause(res.cog_final),
                        "重心_侧面": str(res.cog_side.status),
                        "重心_侧面说明": str(res.cog_side.reason),
                        "重心_侧面原因类型": _cause(res.cog_side),
                        "重心_正面": str(res.cog_front.status),
                        "重心_正面说明": str(res.cog_front.reason),
                        "重心_正面原因类型": _cause(res.cog_front),
                        "回收速度": str(res.retract_speed.status),
                        "回收速度说明": str(res.retract_speed.reason),
                        "回收速度原因类型": _cause(res.retract_speed),
                        "发力顺序": str(res.force_sequence.status),
                        "发力顺序说明": str(res.force_sequence.reason),
                        "发力顺序原因类型": _cause(res.force_sequence),
                        "拳面角度": str(res.wrist_angle.status),
                        "拳面角度说明": str(res.wrist_angle.reason),
                        "拳面角度原因类型": _cause(res.wrist_angle),
                    }
                    if res.cog_com is not None:
                        row["重心_CoM(方案3)"] = str(res.cog_com.status)
                        row["重心_CoM说明"] = str(res.cog_com.reason)
                        row["重心_CoM原因类型"] = _cause(res.cog_com)
                    else:
                        row["重心_CoM(方案3)"] = "未评估"
                        row["重心_CoM说明"] = ""
                        row["重心_CoM原因类型"] = ""
                    rows.append(row)

                    detail_lines = [
                        f"视频: {v.name}",
                        f"视角模式: {res.view_mode}",
                        _fmt_indicator("重心(侧面优先)", res.cog_final),
                        _fmt_indicator("重心_侧面", res.cog_side),
                        _fmt_indicator("重心_正面", res.cog_front),
                        _fmt_indicator("重心_CoM(方案3)", res.cog_com),
                        _fmt_indicator("回收速度", res.retract_speed),
                        _fmt_indicator("发力顺序", res.force_sequence),
                        _fmt_indicator("拳面角度", res.wrist_angle),
                    ]
                    self._set_detail("\n".join(detail_lines))

                    if full_f is not None and meta is not None:
                        payload = {
                            "video_path": str(v),
                            "pose_variant": str(pose_variant),
                            "fps": float(res.fps),
                            "view_mode": str(res.view_mode),
                            "front_segment": None if res.front_segment is None else [int(res.front_segment[0]), int(res.front_segment[1])],
                            "side_segment": None if res.side_segment is None else [int(res.side_segment[0]), int(res.side_segment[1])],
                            "meta": meta,
                            "cog_side": to_jsonable(res.cog_side),
                            "cog_front": to_jsonable(res.cog_front),
                            "cog_final": to_jsonable(res.cog_final),
                            "cog_com": to_jsonable(res.cog_com),
                            "retract_speed": to_jsonable(res.retract_speed),
                            "force_sequence": to_jsonable(res.force_sequence),
                            "wrist_angle": to_jsonable(res.wrist_angle),
                        }
                        full_f.write(json.dumps(payload, ensure_ascii=False) + "\n")

                    if want_debug and lm is not None and vs is not None and meta is not None:
                        out_mp4 = debug_dir / f"{v.stem}_debug.mp4"
                        export_debug_video(
                            v,
                            out_mp4,
                            pose_variant=pose_variant,
                            stance=stance,
                            view_hint=view_hint,
                            res=res,
                            landmarks=lm,
                            view_scores=vs,
                            meta=meta,
                        )

                    done += 1
                    self._progress("评估中", done, len(videos))

                if rows:
                    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                        w.writeheader()
                        w.writerows(rows)

                summary = {
                    "out_dir": str(out_dir),
                    "tech_report_csv": str(csv_path),
                    "tech_report_full_jsonl": str(full_path) if want_full else None,
                    "debug_videos": str(debug_dir) if want_debug else None,
                    "videos_total": int(len(videos)),
                    "videos_done": int(done),
                    "stopped": bool(self._stop_evt.is_set()),
                }
                self._set_raw(json.dumps(summary, ensure_ascii=False, indent=2))
                self._set_result(f"输出目录：{out_dir}")
                self._set_status("评估完成" if not self._stop_evt.is_set() else "已停止（当前视频评估结束后生效）")
            except Exception as e:
                self._set_status("评估失败")
                self._set_result(str(e))
            finally:
                try:
                    if full_f is not None:
                        full_f.close()
                except Exception:
                    pass
                self._set_buttons(False)

        self._worker = threading.Thread(target=_run, daemon=True)
        self._worker.start()

    def _stop(self) -> None:
        self._stop_evt.set()
        self._set_status("正在停止…")

    def _on_close(self) -> None:
        self._stop_evt.set()
        self._win.destroy()

@dataclass(frozen=True)
class UiState:
    source: str
    pose_variant: str
    pose_variant: str
    workers: int
    save_output: bool
    out_path: str | None


class App:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("MediaPipe 动作识别（人体姿态 + 手部）")
        self.root.geometry("1100x720")

        self.source_var = StringVar(value="0")
        self.pose_var = StringVar(value="full")
        self.workers_var = IntVar(value=2)
        self.save_var = BooleanVar(value=False)
        self.out_var = StringVar(value="")
        self.status_var = StringVar(value="就绪")
        self.actions_var = StringVar(value="-")
        self.progress_var = DoubleVar(value=0.0)
        self.progress_text_var = StringVar(value="")

        self._stop_evt = threading.Event()
        self._worker: threading.Thread | None = None
        self._queue: Queue[tuple[np.ndarray, str]] = Queue(maxsize=1)
        self._photo: ImageTk.PhotoImage | None = None
        self._compare_win: CompareWindow | None = None
        self._tech_eval_win: TechEvalWindow | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick()

    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=0)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        # Left: controls
        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))

        card = ttk.Labelframe(left, text="输入", padding=10)
        card.pack(fill="x", pady=(0, 10))

        ttk.Label(card, text="输入源（摄像头编号 或 视频文件路径）：").pack(anchor="w")
        src_row = ttk.Frame(card)
        src_row.pack(fill="x", pady=(6, 0))
        self.source_entry = ttk.Entry(src_row, textvariable=self.source_var)
        self.source_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(src_row, text="选择视频…", command=self._browse_video).pack(side="left", padx=(8, 0))

        opts = ttk.Labelframe(left, text="选项", padding=10)
        opts.pack(fill="x", pady=(0, 10))

        ttk.Label(opts, text="人体姿态模型：").pack(anchor="w")
        ttk.Combobox(opts, textvariable=self.pose_var, values=["lite", "full", "heavy"], state="readonly").pack(
            fill="x", pady=(6, 0)
        )

        ttk.Label(opts, text="离线线程数（视频文件）：").pack(anchor="w", pady=(10, 0))
        self.workers_spin = ttk.Spinbox(opts, from_=1, to=16, textvariable=self.workers_var, width=6)
        self.workers_spin.pack(anchor="w", pady=(6, 0))

        ttk.Checkbutton(opts, text="导出结果视频", variable=self.save_var, command=self._toggle_out).pack(
            anchor="w", pady=(10, 0)
        )
        out_row = ttk.Frame(opts)
        out_row.pack(fill="x", pady=(6, 0))
        self.out_entry = ttk.Entry(out_row, textvariable=self.out_var, state="disabled")
        self.out_entry.pack(side="left", fill="x", expand=True)
        self.out_btn = ttk.Button(out_row, text="选择保存位置…", command=self._choose_out, state="disabled")
        self.out_btn.pack(side="left", padx=(8, 0))

        runbox = ttk.Labelframe(left, text="运行", padding=10)
        runbox.pack(fill="x", pady=(0, 10))
        btn_row = ttk.Frame(runbox)
        btn_row.pack(fill="x")
        self.start_btn = ttk.Button(btn_row, text="开始", command=self._start)
        self.start_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(btn_row, text="停止", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", fill="x", expand=True, padx=(8, 0))

        ttk.Button(runbox, text="动作比对…", command=self._open_compare).pack(fill="x", pady=(10, 0))
        ttk.Button(runbox, text="直拳检测…", command=self._open_tech_eval).pack(fill="x", pady=(8, 0))

        info = ttk.Labelframe(left, text="状态", padding=10)
        info.pack(fill="x")
        ttk.Label(info, textvariable=self.status_var, wraplength=320).pack(anchor="w")
        ttk.Label(info, text="识别结果：").pack(anchor="w", pady=(8, 0))
        ttk.Label(info, textvariable=self.actions_var, wraplength=320).pack(anchor="w")
        ttk.Label(info, textvariable=self.progress_text_var, wraplength=320).pack(anchor="w", pady=(8, 0))
        self.progress_bar = ttk.Progressbar(info, orient="horizontal", mode="determinate", maximum=100.0)
        self.progress_bar.pack(fill="x", pady=(6, 0))
        ttk.Label(info, text="提示：点击“停止”结束识别。").pack(anchor="w", pady=(8, 0))

        # Right: preview
        right = ttk.Labelframe(outer, text="预览", padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.preview = ttk.Label(right)
        self.preview.grid(row=0, column=0, sticky="nsew")

    def _open_compare(self) -> None:
        if self._compare_win and self._compare_win.is_open():
            self._compare_win.focus()
            return
        self._compare_win = CompareWindow(self.root)

    def _open_tech_eval(self) -> None:
        if self._tech_eval_win and self._tech_eval_win.is_open():
            self._tech_eval_win.focus()
            return
        self._tech_eval_win = TechEvalWindow(self.root)

    def _toggle_out(self) -> None:
        enabled = bool(self.save_var.get())
        self.out_entry.configure(state="normal" if enabled else "disabled")
        self.out_btn.configure(state="normal" if enabled else "disabled")
        if not enabled:
            self.out_var.set("")

    def _browse_video(self) -> None:
        p = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4;*.avi;*.mov;*.mkv"), ("所有文件", "*.*")],
        )
        if p:
            self.source_var.set(p)

    def _choose_out(self) -> None:
        p = filedialog.asksaveasfilename(
            title="保存输出视频",
            defaultextension=".mp4",
            filetypes=[("MP4 视频", "*.mp4"), ("AVI 视频", "*.avi"), ("所有文件", "*.*")],
        )
        if p:
            self.out_var.set(p)

    def _collect_state(self) -> UiState:
        source = self.source_var.get().strip()
        if not source:
            raise ValueError("请输入摄像头编号（例如 0）或选择一个视频文件。")

        workers = int(self.workers_var.get() or 1)
        if workers < 1:
            workers = 1

        out_path = None
        if self.save_var.get():
            out_path = self.out_var.get().strip()
            if not out_path:
                raise ValueError("请先选择输出视频保存路径。")

        return UiState(
            source=source,
            pose_variant=self.pose_var.get().strip() or "full",
            workers=workers,
            save_output=bool(self.save_var.get()),
            out_path=out_path,
        )

    def _start(self) -> None:
        if self._worker and self._worker.is_alive():
            return

        try:
            state = self._collect_state()
        except Exception as e:
            messagebox.showerror("配置错误", str(e))
            return

        self._stop_evt.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("启动中…（首次运行可能需要下载模型）")
        self.actions_var.set("-")
        self.progress_var.set(0.0)
        self.progress_text_var.set("")
        self.progress_bar.configure(mode="determinate", maximum=100.0, value=0.0)

        self._worker = threading.Thread(target=self._worker_loop, args=(state,), daemon=True)
        self._worker.start()

    def _stop(self) -> None:
        self._stop_evt.set()
        self.stop_btn.configure(state="disabled")
        self.status_var.set("正在停止…")

    def _on_close(self) -> None:
        self._stop_evt.set()
        self.root.after(50, self.root.destroy)

    def _worker_loop(self, state: UiState) -> None:
        source = state.source
        is_file = not source.isdigit()

        if source.isdigit():
            cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            self._post_status(f"无法打开输入源：{source}")
            self._post_done()
            return

        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps_for_ts = src_fps if (is_file and src_fps > 1e-3) else 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if is_file else 0

        writer: cv2.VideoWriter | None = None
        if state.out_path:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(state.out_path, fourcc, fps_for_ts, (w, h))

        if is_file and state.workers > 1:
            self._worker_loop_parallel_video(state, cap, writer, fps_for_ts, total)
            return

        try:
            models_dir = Path(__file__).resolve().parent / "models"
            pipe = MediaPipePipeline(
                models_dir=models_dir,
                cfg=PipelineConfig(pose_variant=state.pose_variant, running_mode="video"),
            )
        except Exception as e:
            cap.release()
            if writer is not None:
                writer.release()
            self._post_status(f"初始化失败：{e}")
            self._post_done()
            return

        t0 = time.monotonic()
        frame_count = 0
        self._post_status("运行中…")
        self._post_progress(0, total)

        while not self._stop_evt.is_set():
            ok, frame = cap.read()
            if not ok:
                # Video ended or camera read failed.
                self._stop_evt.set()
                break

            ts = pipe.next_timestamp_ms(is_file=is_file, fps_for_ts=fps_for_ts)
            annotated, actions = pipe.annotate(frame, timestamp_ms=ts)

            frame_count += 1
            if writer is not None and is_file and total > 0 and (frame_count % 5 == 0 or frame_count == total):
                self._post_progress(frame_count, total)
            fps = frame_count / max(1e-6, (time.monotonic() - t0))
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                "点击“停止”结束",
                (10, annotated.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(annotated)

            if actions:
                actions_text = ", ".join(ACTION_LABELS_ZH.get(a, a) for a in actions)
            else:
                actions_text = "-"
            self._post_frame(annotated, actions_text)

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        self._post_status("已停止")
        self._post_progress(frame_count, total)
        self._post_done()

    def _worker_loop_parallel_video(
        self,
        state: UiState,
        cap: cv2.VideoCapture,
        writer: cv2.VideoWriter | None,
        fps_for_ts: float,
        total: int,
    ) -> None:
        # Parallel processing for offline videos using IMAGE mode pipelines.
        models_dir = Path(__file__).resolve().parent / "models"
        workers = max(1, int(state.workers))
        frame_q: "Queue[tuple[int, object] | None]" = Queue(maxsize=workers * 2)
        result_q: "Queue[tuple[int, object, list[str]]]" = Queue(maxsize=workers * 2)

        def reader() -> None:
            idx = 0
            while not self._stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                frame_q.put((idx, frame))
                idx += 1
            for _ in range(workers):
                frame_q.put(None)

        def worker() -> None:
            pipe = MediaPipePipeline(
                models_dir=models_dir,
                cfg=PipelineConfig(pose_variant=state.pose_variant, running_mode="image"),
            )
            while True:
                item = frame_q.get()
                if item is None:
                    break
                if self._stop_evt.is_set():
                    continue
                idx, frame = item
                annotated, actions = pipe.annotate(frame, timestamp_ms=None)
                result_q.put((idx, annotated, actions))

        self._post_status(f"运行中…（离线多线程：{workers}）")
        self._post_progress(0, total)

        t_reader = threading.Thread(target=reader, daemon=True)
        t_reader.start()
        worker_ts = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
        for t in worker_ts:
            t.start()

        next_idx = 0
        pending: dict[int, tuple[object, list[str]]] = {}
        written = 0
        t0 = time.monotonic()

        while not self._stop_evt.is_set():
            if next_idx in pending:
                annotated, actions = pending.pop(next_idx)
                written += 1

                if writer is not None:
                    writer.write(annotated)

                if total > 0 and (written % 5 == 0 or written == total):
                    self._post_progress(written, total)

                fps = written / max(1e-6, (time.monotonic() - t0))
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if actions:
                    actions_text = ", ".join(ACTION_LABELS_ZH.get(a, a) for a in actions)
                else:
                    actions_text = "-"
                self._post_frame(annotated, actions_text)
                next_idx += 1
                continue

            try:
                idx, annotated, actions = result_q.get(timeout=0.2)
            except Exception:
                alive = t_reader.is_alive() or any(t.is_alive() for t in worker_ts)
                if (not alive) and (not pending):
                    break
                continue

            pending[int(idx)] = (annotated, actions)

        self._stop_evt.set()
        t_reader.join(timeout=2.0)
        for t in worker_ts:
            t.join(timeout=2.0)

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        self._post_status("已停止")
        self._post_progress(written, total)
        self._post_done()

    def _post_frame(self, frame_bgr: np.ndarray, actions: str) -> None:
        # Keep only the latest frame.
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break
        self._queue.put((frame_bgr, actions))

    def _post_status(self, text: str) -> None:
        # Tkinter updates must happen on the main thread.
        self.root.after(0, lambda: self.status_var.set(text))

    def _post_progress(self, done: int, total: int) -> None:
        def _set() -> None:
            if total > 0:
                self.progress_bar.configure(mode="determinate", maximum=float(total))
                self.progress_bar["value"] = float(done)
                pct = (done / total) * 100.0
                self.progress_text_var.set(f"进度：{done}/{total}（{pct:.1f}%）")
            else:
                # Unknown total (camera): show nothing.
                self.progress_bar.configure(mode="determinate", maximum=100.0)
                self.progress_bar["value"] = 0.0
                self.progress_text_var.set("")

        self.root.after(0, _set)

    def _post_done(self) -> None:
        def _done() -> None:
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")

        self.root.after(0, _done)

    def _tick(self) -> None:
        try:
            frame_bgr, actions = self._queue.get_nowait()
        except Empty:
            self.root.after(30, self._tick)
            return

        self.actions_var.set(actions)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Fit-to-window preview (keep aspect ratio).
        pw = max(1, self.preview.winfo_width())
        ph = max(1, self.preview.winfo_height())
        iw, ih = img.size
        scale = min(pw / iw, ph / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        img = img.resize((nw, nh), Image.BILINEAR)

        self._photo = ImageTk.PhotoImage(img)
        self.preview.configure(image=self._photo)
        self.root.after(30, self._tick)


def main() -> None:
    root = Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
