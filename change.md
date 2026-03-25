# 修改日志

## 2026-01-23: CompareWindow UI 优化

### 问题描述
`CompareWindow` 界面存在以下问题：
- 四个 Labelframe 区块并列，操作流程不清晰
- 模板设置区域控件过多（row2 有6个控件挤在一行）
- 结果展示与进度混在一起，不够直观
- 原始 JSON 数据区域始终显示，占用空间
- **文件选择对话框关闭后，子窗口会跑到主窗口后面**

### 修改内容

**文件**: `app_ui.py:28-540`

#### 1. 添加可折叠面板组件
新增 `CollapsibleSection` 类，支持展开/收起内容区域，用于高级选项和 JSON 数据显示。

#### 2. 步骤式布局重构
将界面重组为三个清晰步骤 + 结果区域：
- **① 准备模板**: 支持「使用已有模板」或「从视频生成」两种模式切换
- **② 选择待比对视频**: 目标视频选择 + 预览导出选项
- **③ 执行比对**: 开始/停止按钮，进度条仅在运行时显示
- **结果**: 大号彩色相似度分数 + 匹配信息 + 可折叠 JSON 详细数据

#### 3. 模板模式切换
- 使用 Radiobutton 让用户选择「使用已有模板」或「从视频生成」
- 根据选择动态显示相关控件，简化界面

#### 4. 高级选项折叠
将 Pose模型、线程数、起始/结束帧 放入可折叠的「高级选项」面板，默认收起。

#### 5. 结果展示优化
- 大号相似度分数显示（36pt 粗体）
- 颜色编码：绿色(≥80%) / 黄色(50-80%) / 红色(<50%)
- 进度条与结果分离：进度条仅在运行时显示
- JSON 数据默认收起，点击展开

#### 6. 修复文件对话框后子窗口焦点问题
- 为所有文件对话框添加 `parent=self._win` 参数
- 对话框关闭后调用 `lift()` 和 `focus_force()` 将子窗口置顶

### 验证方法
```powershell
# 语法检查
.\.venv\Scripts\python.exe -m py_compile .\app_ui.py

# 功能测试
.\.venv\Scripts\python.exe app_ui.py
# 打开「动作比对…」窗口，验证：
# - 步骤式布局显示正确
# - 模板模式切换正常
# - 折叠面板工作正常
# - 比对完成后分数带颜色显示
```

## 2026-01-23: 修复视频比对相似度为0%问题

### 问题描述
进行视频比对时，相同视频的相似度显示为0%（或接近0%）。

### 修改内容

#### 1. 改进分数计算公式
**文件**: `action_compare.py:288-290`, `match_template.py:75-77`

将指数衰减公式改为基线归一化：
```python
# 原公式 (过于敏感):
# score = float(np.exp(-avg_cost))

# 新公式 (基线归一化):
baseline = 2.0
score = float(baseline / (baseline + avg_cost))
```

新公式分数分布：
- 完全匹配 (avg_cost=0): 100%
- 轻微差异 (avg_cost=1): ~67%
- 中等差异 (avg_cost=2): 50%
- 较大差异 (avg_cost=5): ~29%

#### 2. 修复递归调用缺少normalizer参数的bug
**文件**: `action_compare.py:102`

在 workers > 1 但无法获取帧数时的回退路径中添加了缺失的 `normalizer=normalizer` 参数。

### 验证方法
```powershell
# 语法检查
.\.venv\Scripts\python.exe -m py_compile .\action_compare.py .\match_template.py

# 创建模板并比对测试
.\.venv\Scripts\python.exe make_template.py --video test.mp4 --pose heavy
.\.venv\Scripts\python.exe match_template.py --template templates\test_heavy.npz --video test.mp4
```

## 2026-01-23: Dual-template (front+side) scoring + repetition aggregation

### Issue
Need a more stable similarity score for: standard front video + standard side video vs a student's single long video
containing both views. Output should be an integer percent (0-100), and average across repetitions (drop max & min).

### Changes
- `pose_features.py`: add `pose_view_score()` (frontness heuristic) and `mirror_pose_features()` (left/right flip tolerance)
- `action_compare.py`: extract per-frame view scores during feature extraction; auto split front/side segments; select a
  representative repetition from standard templates; multi-match subsequence DTW; aggregate with trimmed mean; add
  `compare_video_to_dual_templates()`
- `DUAL_TEMPLATE_COMPARE.md`: document the pipeline and usage

### Verification
```powershell
.\.venv\Scripts\python.exe -m py_compile .\action_compare.py .\pose_features.py
```

## 2026-01-23: 批量提取正/侧面骨架并分别对比（目录模式）

### 问题描述
需要从「标准样本」中提取正/侧面骨架信息；「学员样本」中的视频包含正+侧两个视角，需要自动拆分并分别与标准正/侧模板进行对比，输出可落地的结果文件。

### 修改内容
- 新增 `batch_dual_compare.py`: 目录批处理脚本
  - 自动识别标准正/侧视频（也可用 `--front/--side` 显式指定）
  - 自动生成标准模板（`.npz`）
  - 对学员目录下所有视频执行 `compare_video_to_dual_templates()`，输出 `compare_results.csv/jsonl`
  - 可选 `--export_raw` 导出原始 Pose33 骨架（`landmarks[T,33,4]`）到 `.npz`，并按 front/side 分段保存
- 更新 `.gitignore`: 忽略 `outputs/`（避免误提交批处理产物）

### 验证方法
```powershell
.\.venv\Scripts\python.exe -m py_compile .\batch_dual_compare.py
.\.venv\Scripts\python.exe .\batch_dual_compare.py --standard_dir "C:\Users\21240\Desktop\标准样本" --student_dir "C:\Users\21240\Desktop\学员样本" --pose full --out_dir "C:\Users\21240\Desktop\对比输出" --export_raw
```

## 2026-01-23: 修复侧面视角分数异常偏低（归一化 v3）

### 问题描述
侧面相似度普遍异常偏低（~0.1），导致综合分很低。怀疑原因是侧面视角下肩宽投影很小、且随人物朝向(yaw)变化明显，使用「肩宽做尺度 + 肩线水平旋转」会放大噪声与朝向差异。

### 修改内容
- `pose_features.py`: 新增 `normalize_pose_xy_v3()`：
  - 尺度优先使用「躯干长度(肩中心-髋中心)」，降低对 yaw 的敏感性
  - 旋转策略改为“正面用肩线水平对齐、侧面用躯干竖直对齐”（更稳定）
- `action_compare.py` / `make_template.py`：模板默认改用 v3 并写入 `feature_layout=..._v3`
- `action_compare.py` / `match_template.py`：根据 `feature_layout` 自动选择 v1/v2/v3 对应 normalizer

### 验证方法
```powershell
.\.venv\Scripts\python.exe -m py_compile .\pose_features.py .\action_compare.py .\make_template.py .\match_template.py

# 样本目录批量跑一遍（生成 v3 模板）
.\.venv\Scripts\python.exe .\batch_dual_compare.py --standard_dir "C:\Users\21240\Desktop\标准样本" --student_dir "C:\Users\21240\Desktop\学员样本" --pose full --out_dir "C:\Users\21240\Desktop\对比输出_v3"
```

## 2026-01-23: 融入得分指南的规则评分

### 问题描述
技术人员提供了得分指南（实战式/直拳），需要在现有正/侧面比对算法上追加规则扣分与明细输出。

### 修改内容
- 新增 `rule_scoring.py`：基于原始 Pose33 关键点的规则引擎，按“违规帧占比”触发扣分并输出明细。
- `action_compare.py`：双模板比对中加入可选规则评分（正/侧分段独立计算），扩展 `DualCompareResult` 字段。
- `batch_dual_compare.py`：增加 `--rules/--action` 参数，CSV/JSONL 输出规则分与扣分合计。
- `DUAL_TEMPLATE_COMPARE.md`：补充规则评分说明与新增字段。

### 验证方法
```powershell
.\.venv\Scripts\python.exe -m py_compile .\action_compare.py .\batch_dual_compare.py .\rule_scoring.py
```

## 2026-01-23: 误差分析方案（A+B）计划文档

### 问题描述
需要基于规则评分与关节误差输出，制定可执行的误差分析方案。

### 修改内容
- 新增 `docs/error_analysis_plan.md`，定义规则层与关节层误差分析的输出、字段与计算逻辑。

### 验证方法
- 目视检查文档内容。

## 2026-01-30: 批处理误差分析输出（规则层 + 关节层）

### 问题描述
需要在不改变现有比对算法分数的前提下，新增“误差分析”产出，帮助定位具体哪条规则、哪个身体部位导致分数偏低。

### 修改内容
- `rule_scoring.py`：扩展规则明细，新增 `valid_frames/total_frames` 字段用于可读性提示。
- `pose_features.py`：新增可返回 DTW 对齐路径的 `subsequence_dtw_with_path()`，用于后续关节误差统计。
- `action_compare.py`：`compare_video_to_dual_templates()` 增加可选关节误差统计（基于 DTW 对齐 + 可见度过滤），并把结果挂到 `DualCompareResult`。
- `batch_dual_compare.py`：
  - 新增 `--error-analysis` 开关；
  - 输出 `error_rules.csv`（需同时开启 `--rules`）与 `error_joints.csv`（关节误差统计）。

### 验证方法
```powershell
.\.venv\Scripts\python.exe -m py_compile .\pose_features.py .\rule_scoring.py .\action_compare.py .\batch_dual_compare.py
```

## 2026-01-30: 直拳技术指标评估补全（重心窗口化 + 发力顺序 + 拳面角度 + 调试视频）

### 问题描述
直拳技术指标评估链路需要按实施计划补齐关键缺口，并保证默认输出“干净”（不带数值），同时提供可排查的完整输出与调试视频能力。

### 修改内容
- `tech_eval.py`
  - 重心：侧面优先，且优先基于“出拳窗口”统计，兜底整段统计
  - 回收速度：默认输出原因不再包含阈值数字（保持干净输出约束）
  - 新增指标：发力顺序（蹬地→转腰→送肩→伸直的可解释代理）与拳面角度（腕-小臂对齐）
  - 新增能力：可导出叠加骨架与指标状态的调试视频 `export_debug_video()`
- `batch_tech_eval.py`
  - CSV 增加“发力顺序/拳面角度”列与原因列
  - JSONL 完整输出同步增加新指标字段
  - 新增 `--debug-video`：输出 `debug_videos/*_debug.mp4`
- `selfcheck_tech_eval.py`
  - 新增自检覆盖：拳面角度与发力顺序，并检查原因文本不含数字
- `docs/直拳_技术指标_剩余任务排期.md`
  - 补充剩余模块优先级与周级执行排期（责任角色/依赖资源/交付物）

### 验证方法
```powershell
.\.venv\Scripts\python.exe -m py_compile .\tech_eval.py .\batch_tech_eval.py .\selfcheck_tech_eval.py
.\.venv\Scripts\python.exe .\selfcheck_tech_eval.py
```

## 2026-01-31: UI 增加直拳检测入口（子界面）

### 问题描述
需要在桌面前端增加“直拳检测”入口按钮，并在子界面中提供目录跑批评估能力，输出 CSV/JSONL 与可选调试视频。

### 修改内容
- `app_ui.py`
  - 新增 `TechEvalWindow` 子界面：选择视频目录、配置 pose/站姿/视角，支持 `tech_report.csv`、可选 `tech_report_full.jsonl`、可选 `debug_videos/*_debug.mp4` 导出
  - 主界面“运行”区域新增按钮 `直拳检测…`，用于打开子界面

### 验证方法
```powershell
.\.venv\Scripts\python.exe -m py_compile .\app_ui.py
.\.venv\Scripts\python.exe app_ui.py
```

## 2026-01-30: 批量提取标准动作骨架数据与骨架视频

### 问题描述
需要从「标准动作视频--分解版」中批量提取每个视频的 Pose33 骨架数据，跳过汇总文件夹，并按动作目录整理输出，同时生成对应的骨架视频与清单。

### 修改内容
- 新增 `batch_export_skeleton.py`：
  - 遍历输入目录并跳过包含“汇总”的子目录
  - 为每个视频导出 Pose33 骨架 `.npz`（含 meta/名称）
  - 同步生成骨架视频 `.mp4`（仅绘制姿态）
  - 输出 `manifest.csv` 汇总清单

### 验证方法
```powershell
.\.venv\Scripts\python.exe .\batch_export_skeleton.py --source_dir "标准动作视频--分解版" --pose full
```

## 2026-01-30: 直拳技术指标（重心偏前/偏后 + 回收速度）批量评估脚本

### 问题描述
需要在单目视频条件下，对直拳训练视频输出“每条技术指标是否合格”的结果。当前优先落地：
- 重心检测：采用方案 1（髋部投影+支撑面）+ 方案 2（膝/踝角规则），侧面优先
- 回收速度：侧面视角检测（伸直到肘角<45°耗时<0.2s），若出拳次数>=4且慢的占比>=75%则判该指标不合格

### 修改内容
- 新增 `tech_eval.py`：
  - Pose33 + view_score 提取
  - 正/侧面分段（保守策略：只有在差异明显时才认为 mixed）
  - 重心（侧面优先）与回收速度指标判定（默认输出不包含杂乱数值）
- 新增 `batch_tech_eval.py`：
  - 批量跑目录内视频，输出干净版 `tech_report.csv`
  - 可选 `--full` 输出 `tech_report_full.jsonl`（包含完整数值/事件明细）

### 验证方法
```powershell
# 语法检查（避免写入 pycache 可加 -B）
.\.venv\Scripts\python.exe -B -m py_compile .\tech_eval.py .\batch_tech_eval.py

# 批量评估（示例）
.\.venv\Scripts\python.exe .\batch_tech_eval.py --video_dir "学员样本" --pose full
.\.venv\Scripts\python.exe .\batch_tech_eval.py --video_dir "学员样本" --pose full --full
```

## 2026-01-30: 补充直拳技术指标实施方案文档

### 问题描述
需要把“直拳技术指标”（发力顺序、拳面角度/轨迹、回收速度、重心、步法协调）落成一份可直接执行的中文 Markdown 文档，并明确输入/输出、阈值、验收方式与风险兜底。

### 修改内容
- 新增 `docs/直拳_重心检测实施计划.md`：包含直拳关键技术指标的实施方案（重心方案1+2+3进阶、回收速度判死口径、发力顺序、拳面角度/轨迹、步法协调待确认）、参数表与验收流程。
  - 已按确认口径更新：轨迹与步法协调“本期不评估”。
  - 侧面/正面视角可通过 `--view auto|side|front` 明确指定，避免单视角被误切。

### 验证方法
- 目视检查文档内容与业务口径一致。




## 2026-01-31: 直拳发力顺序检测算法改进

### 问题描述
发力顺序检测算法存在严重问题：样本视频的蹬腿动作都极其不标准，但检测结果却显示"合格"。

经分析，问题根源在于：
1. **蹬地检测阈值过宽松**：pitch 范围 10°-80° 几乎包含所有脚的状态，dy 阈值 0.02 只需要脚尖比脚跟低一点点就算"蹬地"
2. **缺乏膝盖角度检测**：没有检测后腿膝盖是否参与伸展发力
3. **统计口径问题**：大量检测不到蹬地的事件被标记为 N/A 不计入统计，少数"合格"样本主导了结果

### 修改内容

**文件**: `tech_eval.py:eval_force_sequence()`

#### 1. 收紧静态蹬地检测条件（`_push_off_ok` 函数）
- pitch 范围从 10°-80° 收紧至 30°-60°
- dy 阈值从 0.02 提高至 0.04
- dy_ratio 阈值从 0.4 提高至 0.45
- 新增后腿膝盖角度检测：要求 140°-170°（有明显伸展但不过直）
- 增加关键点可见性检查：需要 heel, toe, knee, ankle 都可见

#### 2. 增加动态蹬地检测
- 新增 `_calc_heel_lift()` 计算脚跟抬高程度
- 新增 `_calc_knee_angle()` 计算后腿膝盖角度
- 当静态检测失败且窗口内有足够帧数(≥5)时，检测脚跟从低到高的动态变化：
  - 后半段比前半段抬高至少 0.10
  - 膝盖角度变化在 -15°~+20° 之间（稳定或略微伸展）

#### 3. 改进判定逻辑
- **新增蹬地检测率统计**：统计检测到蹬地的事件数 / 总事件数
- **增加检测率门槛**：要求标准蹬地检测率 ≥70% 才能给出判定，否则输出"无法判定：蹬地动作检测率不足"
- **提高正确率门槛**：合格阈值从 60% 提高至 70%
- **丰富事件详情**：增加 `push_detected`、`push_dynamic_score` 等字段用于调试

### 验证结果
修改前：2、3、5、6号视频显示"合格 - 发力顺序：多数出拳顺序正确"
修改后：所有6个视频均显示"无法判定 - 发力顺序：标准蹬地检测率不足(X%<70%)，多数出拳蹬地动作不标准或遮挡"

这符合用户反馈"所有视频的蹬腿动作都极其不标准"的实际情况。

### 后续建议
1. **阈值标定**：需要根据标准合格样本重新标定蹬地检测阈值，确保标准动作能被检测到
2. **可视化调试**：建议增加 debug 视频输出，标记检测到的蹬地帧位置，便于人工验证
3. **样本校准**：收集标准蹬地动作的样本，验证算法能否正确识别

### 验证方法
```powershell
# 语法检查
.\.venv\Scripts\python.exe -m py_compile .\tech_eval.py

# 单视频测试
.\.venv\Scripts\python.exe -c "
from tech_eval import evaluate_video
from pathlib import Path
result = evaluate_video(Path('学员样本/1.mp4'), pose_variant='full')
print(f'发力顺序: {result.force_sequence.status} - {result.force_sequence.reason}')
"

# 批量测试
.\.venv\Scripts\python.exe test_force_sequence.py
```

## 2026-01-31: 直拳发力顺序检测算法优化（第二轮）

### 问题描述
使用标准动作视频测试后发现：算法的容差设置和阈值需要进一步调整，以更好地区分标准动作和学员的不标准动作。

### 修改内容

**文件**: `tech_eval.py:eval_force_sequence()`

#### 1. 优化蹬地检测阈值
- pitch 范围：25°-65°（适度收紧）
- dy 阈值：0.03（适度）
- dy_ratio 阈值：0.4
- 膝盖角度：130°-175°

#### 2. 优化动态检测阈值
- lift_increase 阈值：0.08
- late_lifts 中位数阈值：0.20
- knee_change 范围：-20°~+25°

#### 3. 增加顺序判定容差
- 允许蹬地略晚于转腰最多3帧（约0.1秒@30fps），考虑到侧面视角下蹬地信号可能被延迟检测

#### 4. 调整判定门槛
- 蹬地检测率门槛：30%（低于此值输出"无法判定"）
- 顺序正确率门槛：70%（达到此值才算"合格"）

### 验证结果

**标准动作视频（直拳 右侧）:**
```
状态: 合格 - 发力顺序：顺序正确率100%
蹬地检测数: 2/4 (50%)
```

**学员样本:**
```
1.mp4: 合格 - 顺序正确率80%
2.mp4: 合格 - 顺序正确率80%
3.mp4: 不合格 - 顺序正确率60%
4.mp4: 合格 - 顺序正确率100%
5.mp4: 合格 - 顺序正确率75%
6.mp4: 合格 - 顺序正确率100%
```

### 关键发现
1. 侧面视角下，后腿蹬地的检测率普遍不高（30%-60%）
2. 标准动作视频的蹬地和转腰几乎是同时发生的，需要一定容差
3. 3号样本的正确率低于70%阈值，被正确识别为"不合格"

### 后续建议
1. **收集更多标准样本**：用于进一步标定阈值
2. **双视角验证**：考虑同时使用正面和侧面视角来验证发力顺序
3. **可视化调试**：增加 debug 视频标记，显示每个信号的检测时刻

### 验证方法
```powershell
# 语法检查
.\.venv\Scripts\python.exe -m py_compile .\tech_eval.py

# 标准视频测试
.\.venv\Scripts\python.exe test_standard_video.py

# 学员样本测试
.\.venv\Scripts\python.exe test_force_sequence.py
```

## 2026-01-31: 基于NPZ数据分析优化发力顺序检测阈值

### 问题描述
需要根据标准动作的NPZ骨架数据来科学标定蹬地检测阈值，而不是凭经验设置。

### 数据分析
分析了 `outputs/标准动作视频--分解版_骨架_20260130_184655/直拳` 目录下的6个标准动作NPZ文件：

**侧面视角（右侧）关键统计数据：**
| 指标 | P20 | P50 | P80 | 建议阈值 |
|------|-----|-----|-----|----------|
| Pitch (脚底角度) | 21° | 27° | 35° | 20-45° |
| dy_ratio (脚跟抬高比) | 0.37 | 0.52 | 0.68 | ≥0.5 |
| 膝盖角度 | 126° | 146° | 152° | 120-165° |

**关键发现：**
1. 侧面视角下右脚（后腿）的数据最可靠
2. 左侧视角和正面视角下后腿可见性差，数据不可靠
3. 即使是标准动作，蹬地检测率也只有约50%（受限于侧面视角后腿可见性）

### 修改内容

**文件**: `tech_eval.py:eval_force_sequence()`

#### 1. 优化静态蹬地检测阈值
```python
# 基于标准动作NPZ数据分析优化
pitch_ok = 20.0 <= pitch <= 45.0  # 原: 25-65°
dy_ratio_threshold = 0.5           # 原: 0.4
knee_range = (120.0, 165.0)       # 原: 130-175°
```

#### 2. 优化动态检测阈值
```python
lift_increase >= 0.10 and late_lifts_median >= 0.40  # 原: 0.08, 0.20
knee_change in (-15.0, 20.0)                        # 原: -20, 25
```

#### 3. 增加pre窗口
```python
pre = int(round(0.3 * fps))  # 0.3s (原: 0.2s)，给蹬地检测更长窗口
```

#### 4. 调整判定门槛
```python
push_detection_rate < 0.25  # 无法判定 (原: 0.30)
accuracy >= 0.70            # 合格门槛 (保持)
```

### 验证结果

**标准动作视频（直拳 右侧）:**
```
状态: 合格 - 发力顺序：顺序正确率100%
蹬地检测数: 2/4 (50%)
```

**学员样本:**
```
1.mp4: 不合格 - 顺序正确率25%
2.mp4: 合格 - 顺序正确率100%
3.mp4: 无法判定 - 检测率22%<25%
4.mp4: 不合格 - 顺序正确率0%
5.mp4: 不合格 - 顺序正确率60%
6.mp4: 合格 - 顺序正确率75%
```

### 分析工具
新增 `analyze_standard_npz.py`：自动分析标准动作NPZ文件，输出统计数据和阈值建议。

### 验证方法
```powershell
# 数据分析
.\.venv\Scripts\python.exe analyze_standard_npz.py

# 标准视频测试
.\.venv\Scripts\python.exe test_standard_video.py

# 学员样本测试
.\.venv\Scripts\python.exe test_force_sequence.py
```

## 2026-01-31: 发力顺序检测算法最终优化（解决2.mp4误判问题）

### 问题描述
2.mp4的蹬地动作明显不标准（脚左右摇摆），但之前显示"合格 - 100%正确率"。

问题根源：**检测率过低（33%）但正确率100%**
- 共9个出拳事件，只检测到3个蹬地
- 检测到的3个中有2个正确（100%正确率）
- 算法只看正确率，忽略了大量事件检测不到蹬地的问题

### 解决方案
将判定逻辑改为：**检测率<40%直接判不合格**

```python
# 原逻辑（错误）:
# 检测率>=25% + 正确率>=70% => 合格

# 新逻辑（正确）:
# 检测率<40% => 不合格（大量出拳蹬地动作不标准）
# 检测率>=40% + 正确率>=70% => 合格
# 检测率>=40% + 正确率<70% => 不合格
```

理由：侧面视角下虽然后腿可能部分遮挡，但**40%以上的检测率是基本要求**。如果检测率<40%，说明多数出拳没有标准蹬地动作（或动作幅度太小/不规范）。

### 验证结果

**标准动作视频：**
```
✓ 合格 - 检测率50%（2/4），正确率100%
```

**学员样本：**
```
1.mp4: ✗ 不合格 - 顺序正确率25%
2.mp4: ✗ 不合格 - 检测率33%<40% ← 正确识别！
3.mp4: ✗ 不合格 - 检测率22%<40%
4.mp4: ✗ 不合格 - 顺序正确率0%
5.mp4: ✗ 不合格 - 顺序正确率60%
6.mp4: ✓ 合格 - 检测率44%，正确率75%
```

### 后续建议
1. **增加脚左右摇摆检测**：在正面视角下检测两脚距离变化
2. **多视角融合**：结合正面和侧面视角综合判定蹬地动作
3. **增加动作幅度评估**：检测蹬地时后腿膝盖伸展的幅度

### 验证方法
```powershell
# 标准视频测试
.\.venv\Scripts\python.exe test_standard_video.py

# 学员样本测试（重点检查2.mp4）
.\.venv\Scripts\python.exe test_force_sequence.py

# 详细分析2.mp4
.\.venv\Scripts\python.exe analyze_2mp4.py
```

## 2026-01-31: 发力顺序再优化（上步+正面脚部旋转判死）

### 问题描述
现有发力顺序对“上步”与“正面脚部旋转异常”的关注不足，导致蹬地/顺序结果仍不够稳。

### 修改内容
- `tech_eval.py`：在 `eval_force_sequence()` 中新增
  - 上步检测（前脚上步→后脚跟进）并纳入判定
  - 正面脚部旋转方差判死（触发即判蹬地与顺序不合格）
  - reason 文本避免数字，完整细节写入 `detail`
- `analyze_standard_npz.py`：扩展标准NPZ分析
  - 统计上步位移分位数并给出阈值建议
  - 统计正面脚部旋转方差并给出判死阈值建议

### 阈值依据（来自标准NPZ统计）
- 上步位移阈值：`step_front_min=0.01`, `step_back_min=0.01`
- 正面旋转判死阈值：`rotation_var_thr≈1820 (deg^2)`

### 验证方法
```powershell
# 标准NPZ分析（输出上步/旋转阈值建议）
.\.venv\Scripts\python.exe analyze_standard_npz.py

# 发力顺序样本验证
.\.venv\Scripts\python.exe test_force_sequence.py
```
## 2026-02-01: 学员样本直拳发力顺序检测

### 问题描述
用学员样本检查直拳发力顺序指标，定位每位学员问题。

### 修改内容
- 无代码修改；运行批量评估脚本并生成报告 outputs/tech_eval_20260201_221002/tech_report.csv。

### 验证方法
`powershell
.\\.venv\\Scripts\\python.exe batch_tech_eval.py --video_dir \ 学员样本\ --pose full
`
## 2026-02-01: 学员样本剩余指标评估（拳面角度/回收速度/重心）

### 问题描述
用学员样本跑剩余指标：拳面角度、击打后迅速回收、身体重心稳定性。

### 修改内容
- 无代码修改；运行批量评估脚本并生成报告 outputs/tech_eval_20260201_221614/tech_report.csv。

### 验证方法
`powershell
.\\.venv\\Scripts\\python.exe batch_tech_eval.py --video_dir \ 学员样本\ --pose full
`
## 2026-02-01: 学员样本拳面角度不合格细节导出

### 问题描述
需要给出学员样本拳面角度不合格的详细解释（基于事件统计）。

### 修改内容
- 无代码修改；使用 --full 重新生成含 wrist_angle 事件统计的报告 outputs/tech_eval_20260201_223536/tech_report_full.jsonl。

### 验证方法
`powershell
.\\.venv\\Scripts\\python.exe batch_tech_eval.py --video_dir \ 学员样本\ --pose full --full
`
## 2026-02-01: 批量评估默认重模型 + 多线程

### 问题描述
将批量技术评估默认模型改为 heavy，并支持 10 线程并发后重新跑学员样本。

### 修改内容
- atch_tech_eval.py：默认 pose 改为 heavy，新增 --workers（默认 10），批量评估支持线程并发并保持输出顺序。
- 重新运行：生成报告 outputs/tech_eval_20260201_224137/tech_report.csv。

### 验证方法
`powershell
.\\.venv\\Scripts\\python.exe batch_tech_eval.py --video_dir \ 学员样本\ --pose heavy --workers 10
`
## 2026-02-01: 标准动作NPZ改用heavy重跑

### 问题描述
确认标准动作NPZ为 full 生成后，按要求改用 heavy 重新导出。

### 修改内容
- 无代码修改；使用 batch_export_skeleton 以 heavy 重跑标准动作骨架与 NPZ。
- 新输出目录：outputs/标准动作视频--分解版_骨架_20260201_225132。

### 验证方法
`powershell
.\\.venv\\Scripts\\python.exe batch_export_skeleton.py --source_dir \ 标准动作视频--分解版\ --pose heavy
`
## 2026-02-01: 替换标准动作NPZ旧版输出

### 问题描述
按要求直接用 heavy 版本替换旧版标准动作 NPZ 目录。

### 修改内容
- 删除旧目录 outputs/标准动作视频--分解版_骨架_20260130_184655。
- 将 heavy 重跑目录重命名为旧目录名，保持下游路径不变。

### 验证方法
`powershell
# 目录存在性检查
Get-ChildItem -Path  outputs/标准动作视频--分解版_骨架_20260130_184655
`


## 2026-02-02: 项目文件结构整理

### 问题描述
项目根目录下Python文件过多，结构混乱，不利于维护和扩展。需要将文件按功能分类整理到不同文件夹中。

### 修改内容

#### 1. 创建新的目录结构
```
vision/
├── core/          # 核心模块
├── apps/          # 应用入口
├── batch/         # 批处理工具
├── analysis/      # 分析工具
└── tests/         # 测试文件
```

#### 2. 文件迁移
| 原位置 | 新位置 | 分类 |
|--------|--------|------|
| `vision_pipeline.py` | `core/vision_pipeline.py` | 核心模块 |
| `pose_features.py` | `core/pose_features.py` | 核心模块 |
| `action_compare.py` | `core/action_compare.py` | 核心模块 |
| `rule_scoring.py` | `core/rule_scoring.py` | 核心模块 |
| `video_writer.py` | `core/video_writer.py` | 核心模块 |
| `main.py` | `apps/main.py` | 应用入口 |
| `app_ui.py` | `apps/app_ui.py` | 应用入口 |
| `make_template.py` | `apps/make_template.py` | 应用入口 |
| `match_template.py` | `apps/match_template.py` | 应用入口 |
| `batch_dual_compare.py` | `batch/batch_dual_compare.py` | 批处理工具 |
| `batch_export_skeleton.py` | `batch/batch_export_skeleton.py` | 批处理工具 |
| `batch_tech_eval.py` | `batch/batch_tech_eval.py` | 批处理工具 |
| `tech_eval.py` | `analysis/tech_eval.py` | 分析工具 |
| `selfcheck_tech_eval.py` | `analysis/selfcheck_tech_eval.py` | 分析工具 |
| `analyze_2mp4.py` | `analysis/analyze_2mp4.py` | 分析工具 |
| `analyze_standard_npz.py` | `analysis/analyze_standard_npz.py` | 分析工具 |
| `test_force_sequence.py` | `tests/test_force_sequence.py` | 测试文件 |
| `test_left_side.py` | `tests/test_left_side.py` | 测试文件 |
| `test_standard_video.py` | `tests/test_standard_video.py` | 测试文件 |

#### 3. 更新导入路径
- `apps/*`: `from X` → `from core.X`
- `batch/*`: `from X` / `import X` → `from core.X`
- `analysis/*`: `from X` → `from core.X`
- `tests/*`: `from X` → `from core.X` / `from analysis.X`
- `batch/batch_tech_eval.py`: `from tech_eval` → `from analysis.tech_eval`
- `tests/*.py`: `from tech_eval` → `from analysis.tech_eval`

#### 4. 添加 `__init__.py`
为所有新目录添加 `__init__.py` 使其成为Python包。

#### 5. 更新文档
- `AGENTS.md`: 更新项目结构和命令路径
- `CLAUDE.md`: 更新模块说明和命令路径

### 验证方法
```powershell
# 语法检查
.\.venv\Scripts\python.exe -m py_compile .\apps\main.py .\apps\app_ui.py .\core\vision_pipeline.py .\core\pose_features.py .\core\action_compare.py .\apps\make_template.py .\apps\match_template.py .\core\video_writer.py .\batch\batch_dual_compare.py .\core\rule_scoring.py

# 运行UI
.\.venv\Scripts\python.exe apps/app_ui.py

# 运行CLI
.\.venv\Scripts\python.exe apps/main.py --source 0
```

### 影响说明
- 所有运行命令路径需要更新为新的路径（如 `apps/main.py`）
- IDE的Python路径配置可能需要调整
- 现有脚本调用需要更新路径


## 2026-02-02: 实现重心检测方案3 - 分段质心法 (CoM)

### 问题描述
方案1（髋部代理）在"上身大幅前倾/后仰"场景下会失真，需要实现方案3（分段质心法）作为更鲁棒的重心评估方法。

### 修改内容

#### 1. 新增文件内容 `analysis/tech_eval.py`

**身体分段质量比例**（基于Winter DA. Biomechanics of Human Movement）：
```python
BODY_SEGMENT_MASS_RATIOS = {
    "head": 0.081,
    "trunk": 0.497,
    "upper_arm_l/r": 0.028,
    "forearm_l/r": 0.015,
    "hand_l/r": 0.006,
    "thigh_l/r": 0.100,
    "shank_l/r": 0.047,
    "foot_l/r": 0.014,
}
```

**新增函数**：
- `_compute_segment_center()` - 计算身体段中心点
- `_compute_body_com_single()` - 计算单帧人体质心（CoM）
- `eval_cog_com()` - 基于CoM的重心判定主函数

**CoM计算逻辑**：
1. 将人体分为11段（头、躯干、左右上臂、左右前臂、左右手、左右大腿、左右小腿、左右脚）
2. 根据解剖学质量比例加权
3. 躯干质心偏下40%（肩:髋 = 0.6:0.4）
4. 四肢简化为段中点
5. 至少需要50%身体质量可见

**阈值调整**：
- `com_r_back = 0.38`（比髋部阈值0.35略宽松）
- `com_r_front = 0.62`（比髋部阈值0.65略宽松）

#### 2. 修改 `_eval_cog_side_prefer_punch_windows()`
- 添加 `use_com` 参数控制是否同时返回CoM结果
- 返回类型改为 `tuple[IndicatorResult, IndicatorResult | None]`
- CoM评估同样基于出拳窗口

#### 3. 修改 `TechEvalResult` 数据结构
- 新增字段 `cog_com: IndicatorResult | None` - 方案3评估结果

#### 4. 修改 `_evaluate_from_arrays()`
- 调用 `_eval_cog_side_prefer_punch_windows(use_com=True)`
- 无侧面段时，全段使用CoM评估作为参考

#### 5. 修改 `evaluate_video_full()`
- 输出结果包含 `cog_com` 字段

#### 6. 修改 `export_debug_video()`
- 调试视频叠加显示 `CoM(方案3)` 状态

#### 7. 修改 `batch/batch_tech_eval.py`
- CSV输出新增列：`重心_CoM(方案3)`、`重心_CoM说明`
- JSONL输出包含 `cog_com` 完整数据

### 方案对比

| 特性 | 方案1（髋部代理） | 方案3（CoM分段质心） |
|------|------------------|---------------------|
| **计算基准** | 髋部中心 | 全身质量加权质心 |
| **前倾场景** | 髋部偏后→误判"重心偏后" | CoM前移→正确反映前倾 |
| **后仰场景** | 可能漏判 | 正确检测后仰 |
| **计算复杂度** | O(1) | O(n) |
| **关键点依赖** | 仅需髋部 | 需全身关键点 |

### 输出示例

```json
{
  "cog_final": {"status": "合格", "reason": "侧面：重心居中"},
  "cog_side": {"status": "合格", "reason": "侧面：重心居中"},
  "cog_front": {"status": "无法判定", "reason": "正面段缺失"},
  "cog_com": {
    "status": "合格", 
    "reason": "重心居中（CoM）",
    "detail": {
      "method": "CoM_segment_based",
      "com_ratio_stats": {"mean": 0.48, "std": 0.05, "median": 0.48},
      "valid_frames": 45
    }
  }
}
```

CSV输出：
| 视频 | 重心(侧面优先) | ... | 重心_CoM(方案3) | 重心_CoM说明 |
|------|---------------|-----|----------------|-------------|
| 1.mp4 | 合格 | ... | 合格 | 重心居中（CoM）|

### 验证方法

```powershell
# 语法检查
.\.venv\Scripts\python.exe -m py_compile .\analysis\tech_eval.py .\batch\batch_tech_eval.py

# 运行自检测试
.\.venv\Scripts\python.exe -c "from analysis.tech_eval import eval_cog_com; print('CoM import OK')"

# 批量评估（包含CoM结果）
.\.venv\Scripts\python.exe batch/batch_tech_eval.py --video_dir "学员样本" --out_dir "outputs/com_test" --full

# 检查CSV输出包含"重心_CoM(方案3)"列
```

### 后续建议

1. **阈值标定**：CoM的`r`分布与髋部代理不同，建议用标准合格视频重新标定`0.38/0.62`阈值
2. **对比验证**：并行运行方案1和方案3，收集对比数据验证效果
3. **融合策略**：若两者结果不一致，可设计融合规则（如优先CoM或加权平均）


## 2026-02-02: 修复CoM方向一致性并回归学员样本

### 问题描述
CoM分段质心评估未对齐人物朝向，侧面朝左/朝右时前后判定可能翻转。

### 修改内容
- `analysis/tech_eval.py` 在 `eval_cog_com()` 中引入 `_frame_dir()`，统一前后方向：
  - `com_x` 和脚部支撑边界按朝向做投影（`dir_x`）
  - 无法判定朝向的帧计入 `unknown`，避免误判

### 验证方法
```powershell
# 语法检查
.\.venv\Scripts\python.exe -m py_compile .\apps\main.py .\apps\app_ui.py .\core\vision_pipeline.py .\analysis\tech_eval.py

# 学员样本批量评估（module方式确保import路径正确）
.\.venv\Scripts\python.exe -m batch.batch_tech_eval --video_dir ".\学员样本" --pose heavy --view auto
```

### 验证结果
- 批量报告输出到：`batch\outputs\tech_eval_20260202_114036\tech_report.csv`


## 2026-02-02: 输出指标原因类型规范化

### 问题描述
指标输出仅有结论文本，缺少结构化“原因类型”，不利于统计与定位问题点。

### 修改内容
- `analysis/tech_eval.py` 为各指标补充 `detail.primary_cause`，并在重心类指标增加比例统计（forward/backward/center/unknown）。
- `analysis/tech_eval.py` 为发力顺序增加 `failed_stage`，明确失败环节（rotation/push_off/step/sequence）。
- `batch/batch_tech_eval.py` CSV新增“原因类型”列，并默认使用 `evaluate_video_detail()` 保留详情输出。

### 验证方法
```powershell
# 语法检查
.\.venv\Scripts\python.exe -m py_compile .\analysis\tech_eval.py .\batch\batch_tech_eval.py

# 批量评估（包含原因类型列）
.\.venv\Scripts\python.exe -m batch.batch_tech_eval --video_dir ".\学员样本" --pose heavy --view auto
```

### 验证结果
- 批量报告输出到：`batch\outputs\tech_eval_20260202_115606\tech_report.csv`


## 2026-02-02: UI同步输出原因类型与CoM

### 问题描述
直拳检测 UI 导出的 CSV/JSONL 未包含“原因类型”与 CoM（方案3）字段。

### 修改内容
- `apps/app_ui.py` 批量评估输出新增“原因类型”列，并补充 `重心_CoM(方案3)` 三列。
- `apps/app_ui.py` 默认使用 `evaluate_video_detail()`，确保详情字段可用。
- `apps/app_ui.py` JSONL 输出补充 `cog_com`。
- `apps/app_ui.py` 评估窗口新增“指标详情（当前视频）”，展示每项指标的原因类型与失败环节。

### 验证方法
```powershell
.\.venv\Scripts\python.exe -m py_compile .\apps\app_ui.py
```


## 2026-02-04: 修复模块导入路径错误

### 问题描述
运行 `apps/app_ui.py` 时出现 `ModuleNotFoundError: No module named 'pose_features'` 错误，这是因为 `core/rule_scoring.py` 在2026-02-02项目结构整理后仍使用绝对导入路径。

### 修改内容
- `core/rule_scoring.py`: 修复导入路径
  - `from vision_pipeline import ...` → `from .vision_pipeline import ...`

### 验证方法
```powershell
# 语法检查
.\.venv\Scripts\python.exe -m py_compile apps/app_ui.py apps/main.py apps/make_template.py apps/match_template.py

# 导入测试
.\.venv\Scripts\python.exe -c "from core.action_compare import compare_video_to_template; print('Import successful!')"
```


## 2026-02-04: 修复更多模块导入路径错误

### 问题描述
检查发现有其他 4 个文件在2026-02-02项目结构整理后仍使用绝对导入路径引用 `tech_eval` 模块。

### 修改内容
- `analysis/analyze_2mp4.py`: `from tech_eval import ...` → `from analysis.tech_eval import ...`
- `analysis/selfcheck_tech_eval.py`: `from tech_eval import ...` → `from analysis.tech_eval import ...`
- `analysis/analyze_standard_npz.py`: `from tech_eval import ...` → `from analysis.tech_eval import ...`
- `apps/app_ui.py`: `from tech_eval import ...` → `from analysis.tech_eval import ...`

### 验证方法
```powershell
.\.venv\Scripts\python.exe -m py_compile apps/app_ui.py analysis/analyze_2mp4.py analysis/selfcheck_tech_eval.py analysis/analyze_standard_npz.py
```

### 结果
所有文件语法检查通过，导入路径错误已全部修复。
