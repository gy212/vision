# Vision — 武术散打动作识别与评分系统

基于 MediaPipe 的实时姿态识别系统，使用 DTW（动态时间规整）进行动作模板匹配，支持双模板（正面+侧面）对比评分和规则扣分。

## 功能

- 实时摄像头姿态识别与动作分类（抬手、下蹲、V 手势等）
- 从视频创建标准动作模板（`.npz`）
- 单模板 / 双模板（正面+侧面）动作匹配与评分
- 规则引擎扣分（基于 Pose33 原始关键点）
- 批量目录级对比，输出 CSV/JSONL
- Tkinter 桌面 GUI（中文界面）

## 依赖

- Python 3.10+
- MediaPipe 0.10.31
- OpenCV 4.13
- NumPy / Pillow

## 安装

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows
pip install -r requirements.txt
```

## 使用

### 桌面 GUI

```bash
python apps/app_ui.py
```

### 摄像头实时识别

```bash
python apps/main.py --source 0
```

### 离线视频处理（多线程）

```bash
python apps/main.py --source input.mp4 --pose heavy --out out.mp4 --no-show --workers 4
```

### 创建动作模板

```bash
python apps/make_template.py --video action.mp4 --pose heavy --preview
```

### 模板匹配

```bash
python apps/match_template.py --template template.npz --video test.mp4 --preview
```

### 批量双模板对比

```bash
python batch/batch_dual_compare.py \
  --standard_dir "标准样本" --student_dir "学员样本" \
  --pose full --out_dir "输出目录" --rules --action both
```

## 项目结构

```
core/
  vision_pipeline.py   # MediaPipe 封装，姿态/手势检测
  pose_features.py     # 姿态归一化 + DTW 子序列匹配
  action_compare.py    # 模板创建与对比（含双模板）
  rule_scoring.py      # 规则扣分引擎
  video_writer.py      # 编解码器自适应视频输出
apps/
  app_ui.py            # Tkinter 桌面 GUI
  main.py              # CLI 入口
  make_template.py     # 模板创建工具
  match_template.py    # 模板匹配工具
batch/
  batch_dual_compare.py    # 批量双模板对比
  batch_export_skeleton.py # 批量骨架导出
  batch_tech_eval.py       # 批量技术评估
analysis/                  # 分析与自检脚本
tests/                     # 测试用例
```

## License

MIT
