# PosterAgent 课程汇报 PPT 脚本（中文）

> **汇报人：** 李江涛（23307130167）  
> **课程：** 计算机图形学 Project 3  
> **时长：** 10 分钟（建议 **15 页**）  
> **项目：** PosterAgent — 基于层级多智能体协作的学术海报生成优化系统  
> **代码仓库：** https://github.com/Ljt-lj/Hierarchical-Multi-Agent-Collaborative-Optimization-for-Academic-Poster-Generation

---

## 时间分配总览

| 模块 | 建议时长 | 页数 |
|------|----------|------|
| 开场 | 0:30 | 2 页 |
| 选题与任务 | 1:30 | 2 页 |
| **方法与创新（重点）** | **5:00** | **7 页** |
| 结果展示 | 2:00 | 3 页 |
| 人员分工与总结 | 1:00 | 2 页 |
| **合计** | **10:00** | **15 页** |

> **汇报策略：** 课程中的 GenPilot / Paper2Poster / PosterForest **不在单独章节展开**；在介绍 PosterAgent 时**用 1–2 句话**说明技术渊源即可。节省的时间全部用于 **五大创新** 与 **GRASP 实验**。

---

## Slide 1｜封面（0:15）

**页面内容**
- 标题：基于层级多智能体协作的学术海报生成优化系统
- 副标题：PosterAgent — Training-Free 学术海报自动生成
- 汇报人：李江涛｜23307130167｜复旦大学

**演讲词**
> 各位老师、同学好。我汇报的项目是 PosterAgent——面向 PDF 论文自动生成可读学术海报的系统。

**配图：** 见 [附录：插图清单](#附录插图清单) · Slide 1

---

## Slide 2｜汇报提纲（0:15）

**页面内容**
1. 选题：PDF 论文 → 学术海报，为何选 GRASP
2. PosterAgent 架构与五大创新（**重点**）
3. GRASP 实验结果与人员分工

**演讲词**
> 今天按「问题 → 方法 → 结果」来讲，重点是我们的系统设计和 GRASP 上的效果。

---

# 一、选题与任务（1:30）

---

## Slide 3｜我们要解决什么问题？（0:45）

**页面内容**
- **输入：** PDF 论文 → **输出：** 3600×2400 学术海报（PNG/PPTX）
- **Training-Free：** 不微调 T2I/布局模型，LLM API + PyMuPDF / python-pptx
- **核心难点（海报特有）：**
  - 逻辑链：Methods 与 Results 要对得上
  - 插图：多 panel 论文图不能糊、不能腰斩
  - 版式：三栏语义分区，不是均匀切格子

**技术背景（页脚小字，口述 1 句即可）**  
> 课程中 GenPilot、Paper2Poster、PosterForest 已展示「多 Agent + 迭代优化」的可行性；PosterAgent 在此范式上，针对**学术海报**补齐逻辑规划、语义三栏、原图裁剪与多指标控制。

**演讲词**
> 任务是把二十页论文压成一张可读海报。难点不在写摘要，而在**逻辑、原图、版式**同时成立。我们的实现建立在课程多智能体迭代框架思路上，但面向海报做了专门设计——后面会具体讲。

---

## Slide 4｜案例：GRASP（0:45）

**页面内容**
- Nature Methods 2025，22 页，DockQ benchmark，Fig.1–3 多 panel
- **压力测试点：** Fig.2 整图缩放 → 坐标轴不可读；Methods→Results 需逻辑闭环

**演讲词**
> 选 GRASP 是因为它同时考察长文压缩、复杂配图和三栏结构——能检验系统是否真的可用。

**配图：** 见附录 · Slide 4

---

# 二、PosterAgent 方法与创新（5:00）

---

## Slide 5｜总体架构 + 五大创新一览（0:50）

**页面内容**
- **Phase 1：** PDF → Parser → Raw Tree → LogicPlanner → LogicPlan  
- **Phase 2 迭代环：** Refiner → Expander → Visual → Balancer → Painter → Semantic → Layout → Render → ErrorAnalyzer → Commenter → **Controller** → 最佳 PNG/PPTX  
- **五大创新速览（表格）：**

| # | 创新点 | 一句话 |
|---|--------|--------|
| 1 | Logic-First | LogicPlan + logic_links + SemanticCheck |
| 2 | 语义三栏 | 25-50-25 + SectionExpander + ColumnBalancer |
| 3 | 原图裁剪 | 240DPI 页渲染 + 角色裁剪 + figure_curator |
| 4 | 多指标控制 | 0.4·Sem + 0.35·LB + 0.25·ITM，τ=0.85 早停 |
| 5 | 错误路由 | overflow / 缺图 / 逻辑 → 分流修复 |

**演讲词**
> 整体仍是「解析 → 规划 → 迭代渲染」，但 Phase 1 多了逻辑规划，Phase 2 用 Controller 做多指标早停，ErrorAnalyzer 把问题路由到不同 Agent。下面五页分别展开表中五项。

**配图：** 见附录 · Slide 5（Phase 1 / Phase 2 分图）

---

## Slide 6｜创新 1：Logic-First 逻辑规划（0:50）

**页面内容**
- LogicPlanner 抽取 `pipeline_steps`、`validation_chain`
- Refiner 写 `logic_links`；SemanticCheck 计 logic_score
- **GRASP 例：** RPR/IR 构图 → Evoformer → DockQ 验证，贯穿 Methods→Results

**演讲词**
> 先规划论证链再写 bullet，避免 Results 引用 Methods 里没出现过的概念。

---

## Slide 7｜创新 2：语义三栏 + 内容级列平衡（0:55）

**页面内容**
- 标题驱动 **25% / 50% / 25%**（Intro+Methods | Results | 结语+参考文献）
- SectionExpander：Results → 4 子 panel（benchmark / structure / DockQ / 数值）
- ColumnBalancer：**改 bullet 内容**平衡三栏，不拉行距

**演讲词**
> 按海报阅读习惯把一半宽度给 Results，并用内容级平衡消除大块留白。

**配图：** 见附录 · Slide 7

---

## Slide 8｜创新 3：原图渲染 + 角色裁剪（1:00）★图形学亮点

**页面内容**
- 240 DPI 渲染 figure 页（非小尺寸嵌入图）
- 水平空白带切分 → 按 panel 语义分配（figure_curator）
- contain 显示，单 panel 最高约 42% 区域高度
- **效果：** Fig.2 有效高度约 **2×**；无 panel 中间「腰斩」

**演讲词**
> 把排版问题的一部分转化为图形学裁剪——只展示当前 panel 需要的子图，轴标签才读得清。

**配图：** 见附录 · Slide 8

---

## Slide 9｜创新 4：多指标自适应控制（0:55）

**页面内容**
- overall = **0.4×semantic + 0.35×LB + 0.25×ITM**（semantic 含 logic_score）
- τ=**0.85** 早停，保留**最高分**迭代
- GRASP（**v19**）：**2 轮迭代 + crop**；ITM 0.85 → 0.90；保留最优 Overall **0.808**

**演讲词**
> 不只看 overflow，而是逻辑、版式平衡、图文匹配一起评；达标即停，省算力。

**配图：** 见附录 · Slide 9

---

## Slide 10｜创新 5：错误路由 + 双反馈环（0:50）

**页面内容**
- **PosterErrorAnalyzer 路由：**
  - overflow → LayoutAgent
  - 缺图 / sparsity → VisualAgent / Refiner
  - logic 低 → Refiner + LogicPlan
- **双环：** 外层 Controller 改内容；内层 Renderer 溢出重排

**演讲词**
> 错误分类后精准修复，避免「一锅端」式重生成。

**配图：** 可选 · Slide 10（路由示意，架构图局部放大即可）

---

# 三、结果展示（2:00）

---

## Slide 11｜定量结果（0:40）

**页面内容**

| 迭代 | ITM | logic | **Overall** |
|------|-----|-------|-------------|
| 1 | 0.85 | 0.80 | 0.742 |
| 2 | 0.85 | 1.00 | 0.784 |
| 3（crop） | **0.90** | 0.85 | **0.808** |

- v19 最终版：2 轮 Controller 迭代 + 角色裁剪；保留最高分输出（`v19_crop`）

**演讲词**
> 前两轮主要优化逻辑链与版式；第三轮角色裁剪后 ITM 继续提升，输出 `v19_crop` 作为最终海报。

**配图：** 见附录 · Slide 11

---

## Slide 12｜定性对比：v16 vs v19_crop（0:40）

**页面内容**
- **v16：** 三栏 + balancer，Overall 0.858，Fig.2/3 仍偏小
- **v19_crop：** 角色裁剪后轴标签可读，Results 四 panel 清晰
- 消融：完整 focus 图不可读 → 角色裁剪可读、无腰斩

**演讲词**
> 自动评分在 v19 已稳步提升；肉眼可读性靠 iter3 角色裁剪（`v19_crop`）进一步改善。

**配图：** 见附录 · Slide 12

---

## Slide 13｜最终海报全图（0:40）

**页面内容**
- 全页展示最终海报
- 三处标注：① fig1 intro crop  ② fig2 benchmark crop  ③ 右栏无大块空白

**演讲词**
> 这是 PosterAgent 在 GRASP 上的最终输出：逻辑、版式、插图三点同时成立。

**配图：** 见附录 · Slide 13

---

# 四、人员分工与总结（1:00）

---

## Slide 14｜人员分工（0:30）

**页面内容**
- **李江涛 / 23307130167 — 独立完成**
- 架构与 15 Agent、三栏布局、figure_cropper、GRASP 实验 v8–v19、论文与汇报

**演讲词**
> 个人项目，从 Parser 到裁剪到 Controller 均为本人实现。

---

## Slide 15｜总结 & Q&A（0:30）

**页面内容**
- **一句话：** PosterAgent = 多 Agent 迭代 + **LogicPlan、语义三栏、原图裁剪、多指标早停、错误路由**
- **GRASP v19：** overall 0.74 → 0.81（crop）；最终海报 `v19_crop`
- GitHub 二维码 | 欢迎提问

**演讲词**
> 我们在课程多智能体框架之上，解决了学术海报特有的逻辑、可读配图和版式问题。谢谢！

**配图：** 见附录 · Slide 15（GitHub 二维码，可选）

---

## 附录 A｜演讲节奏备忘

| 时间 | 页码 | 要点 |
|------|------|------|
| 0:30 | Slide 2 | 开场结束 |
| 2:00 | Slide 4 | 选题讲完，进入方法 |
| 2:50 | Slide 5 | 架构 + 五大创新一览 |
| 7:00 | Slide 10 | 创新点讲完 |
| 9:00 | Slide 13 | 海报全图 |
| 10:00 | Slide 15 | 结束 |

**若超时：** 压缩 Slide 6、10 口述；Slide 11 表格只报 Overall / ITM。  
**若提前：** 在 Slide 8、12 多展示裁剪对比。

---

## 附录 B｜插图清单

> 路径均相对于项目根目录 `计算机图形学2.0/`。打 **✓** 表示仓库中已有；**待制** 表示需自行导出或绘制。

| 页码 | 位置建议 | 所需图片 | 推荐文件路径 | 说明 |
|------|----------|----------|--------------|------|
| **1** | 封面背景或右侧大图（可半透明） | 最终海报 | `outputs/Integrating_diverse_experimental_information_to_assist_prote/s41592_02820_v19_crop.png` ✓ | 也可用 `paper/overleaf/figures/poster_final.png` ✓ |
| **2** | 无 / 纯文字 | — | — | 提纲页可不插图 |
| **3** | 右侧示意图（可选） | 输入输出示意 | 自制：PDF 图标 → 海报图标 | 无则用 bullet 即可 |
| **4** | 左：论文首页；右：Fig.2 原图 | GRASP 论文 & Fig.2 | 论文 PDF 截图；`s41592-025-02820-1_images/fig2_composite.png` ✓ 或 `fig2_focus.png` ✓ | 说明「整图缩放会糊」 |
| **5** | **上 Phase 1 / 下 Phase 2** | 两阶段流程图 | 上：`paper/figures/slide5_phase1_flow.png` ✓<br>下：`paper/figures/slide5_phase2_flow.png` ✓ | 完整架构图备用：`posteragent_architecture.png` ✓ |
| **6** | 可选：LogicPlan 片段 | logic_links 示意 | 自制：从 LogicPlan JSON 或海报 Methods→Results 箭头标注 | 非必须 |
| **7** | 左：三栏示意图；右：海报局部 | 三栏布局 + 标注 | 自制三栏框图；最终海报 `s41592_02820_v19_crop.png` ✓ 上标 25-50-25 | 右图圈出 Results 四 panel |
| **8** | **左右对比（重点）** | focus vs crop | 左：`s41592-025-02820-1_images/fig2_focus.png` ✓<br>右：`s41592-025-02820-1_images/fig2_crop_benchmark.png` ✓ | 标题：「整图缩放 vs 角色裁剪」 |
| **9** | **折线图（主图）** | Overall / ITM / logic 随迭代变化 | `paper/figures/slide9_score_chart.png` ✓ | 数据：`s41592_02820_v19_scores.json`（iter3=crop） |
| **10** | 可选 | 错误路由示意 | 架构图 Slide 5 右下角「错误路由」局部放大 ✓ | 可不单独做页 |
| **11** | 图表 + 小表 | 评分折线或柱状 | 同 Slide 9：`paper/figures/slide9_score_chart.png` ✓；表格可直接打在页内 | 与口述数字一致 |
| **12** | **上下或左右对比** | v16 vs v19 海报 | 上/左：`s41592_02820_v16` 对应 PNG（outputs 目录）✓<br>下/右：`s41592_02820_v19_crop.png` ✓ | 若无 v16 图，用 v19 vs `fig2_focus` 作插图对比 |
| **13** | **全页** | 最终海报高清 | `outputs/.../s41592_02820_v19_crop.png` ✓ | 加 3 处标注箭头（fig1/fig2/右栏） |
| **14** | 无 | — | — | 分工页纯文字 |
| **15** | 右下角 | GitHub 二维码 | `paper/figures/github_qrcode.png` ✓ | 扫码跳转仓库首页 |

### 备用素材（未强制绑定某一页）

| 文件路径 | 可用于 |
|----------|--------|
| `s41592-025-02820-1_images/fig1_crop_intro.png` ✓ | Slide 13 标注 intro crop |
| `s41592-025-02820-1_images/fig3_crop_structure.png` ✓ | Slide 12/13 structure panel |
| `s41592-025-02820-1_images/fig3_focus.png` ✓ | Slide 8 补充对比 |
| `outputs/.../s41592_02820_v19_iter*.png` ✓ | Slide 11/12 迭代过程（若需） |

### 待制清单汇总

1. ~~**迭代评分折线图**（Slide 9、11）~~ ✓ `paper/figures/slide9_score_chart.png`（`python tools/generate_slide9_chart.py`）  
2. ~~**GitHub 二维码**（Slide 15）~~ ✓ `paper/figures/github_qrcode.png`  
3. **三栏示意图**（Slide 7，可选）— 三个矩形 25:50:25 即可。

---

*脚本 v3｜基线并入技术背景一句带过｜15 页｜2026-06-02*
