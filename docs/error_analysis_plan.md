# 误差分析方案计划（A+B）

目标：在不改变现有比对算法结果的前提下，新增“误差分析”产出，帮助定位具体哪条规则与哪个身体部位导致分数偏低。

范围：先实现 A 规则层误差分析 + B 关节/肢段误差分析，默认输出到批处理结果目录。

---

## A. 规则层误差分析（Rule Error Analysis）

### 输出文件
- error_rules.csv

### 输出字段
- video：视频文件名
- view：front/side
- rule_id：规则 ID（如 stance_elbow）
- rule_name：规则中文名
- violation_ratio：违规占比
- penalty：扣分（0/规则分）
- valid_frames：有效帧数（可见度满足要求）
- total_frames：该视角分段总帧数

### 计算逻辑
1. 使用现有规则引擎的“违规帧占比”结果。
2. 记录每条规则的有效帧数与违规占比。
3. 只要启用规则评分，默认输出 error_rules.csv。

### 价值
直接回答“哪条规则最拉分”，适合一眼定位问题点。

---

## B. 关节/肢段误差分析（Joint Error Analysis）

### 输出文件
- error_joints.csv

### 输出字段
- video
- view
- joint：关节名称（如 L_SHOULDER）
- mean_dist：平均偏差
- p90_dist：90 分位偏差
- max_dist：最大偏差
- valid_frames

### 计算逻辑
1. 使用模板与学员的 DTW 匹配路径，对齐后对每帧计算关节差异。
2. 关节集合：BlazePose 11..32（去脸部点，和现有特征一致）。
3. 对每个关节统计 mean / p90 / max 偏差。
4. 仅统计可见度足够的帧，记录 valid_frames。

### 价值
回答“哪个部位差”（例如手臂、下肢等），用于动作纠正。

---

## 集成方式

### 新增输出
在批处理输出目录内新增：
- error_rules.csv
- error_joints.csv

### 开关
新增 CLI 参数：
- --error-analysis（默认关闭）
  - 仅开启 A+B 分析，不影响现有相似度输出。

---

## 验证方式

1. 运行批处理并开启误差分析开关：
   .\.venv\Scripts\python.exe .\batch_dual_compare.py --standard_dir ... --student_dir ... --rules --action both --error-analysis
2. 检查输出目录存在 error_rules.csv 与 error_joints.csv。
3. 随机抽取 1 个视频，确认：
   - error_rules.csv 里有规则违规占比与扣分。
   - error_joints.csv 里有多个关节的误差统计值。

---

## 风险与边界

- 如果视角分段不稳定，误差分析可能被“转身帧”污染；可复用现有分段边界收缩逻辑。
- 当姿态检测质量差时，有效帧会很少，统计波动变大；需保留 valid_frames 作为可读性提示。
