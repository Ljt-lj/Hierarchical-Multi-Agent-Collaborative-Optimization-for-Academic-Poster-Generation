# Project 3 实验方案：基于层级多智能体协作的学术海报生成优化系统

> **课程**：2026春 计算机图形学
> **代码仓库**：[GitHub链接待补充]

## 1. 摘要

现有的学术海报自动生成方法（如 Paper2Poster、PosterForest）普遍存在信息逻辑断裂、视觉反馈维度单一、迭代机制僵化等问题。本项目提出一种**无需训练、基于层级多智能体协作**的优化框架，通过引入跨节点语义一致性检查、动态布局权重分配以及多维度的视觉-语义反馈闭环，生成逻辑连贯、视觉平衡的高质量学术海报。在60篇论文上的实验表明，本方法在内容完整性和视觉层次清晰度上均优于基线方法。

## 2. 研究背景与意义

学术海报是会议展示的重要载体，但手动排版耗时费力。当前的自动化方案存在以下痛点：

1. **逻辑碎片化**：长文本压缩时丢弃了章节间的依赖关系（如“实验”与“方法”不匹配）。
2. **反馈单一**：仅检测文本溢出或留白，忽略了图文语义匹配度。
3. **迭代僵化**：预设固定迭代次数（如2轮），要么浪费算力，要么优化不足。

本实验基于课件中提到的 GenPilot 和 PosterForest 思想，扩展了**动态评分驱动**的优化机制，旨在解决上述问题。

## 3. 相关工作

| 方法 | 核心思路 | 局限性 |
|:---|:---|:---|
| **Paper2Poster** | Parser-Planner-Painter 流水线 | 缺乏跨节点语义检查，反馈机制简单 |
| **PosterForest** | 层级树结构联合优化 | 迭代次数固定，未考虑内容权重 |
| **GenPilot** | 错误分析与提示词优化 | 主要针对文生图，未适配2D布局任务 |

## 4. 实验方案

### 4.1 系统架构

我们将构建一个包含8个智能体的协作系统，流程如下：

```mermaid
graph TD

A[输入: PDF论文] --> B[解析智能体<br/>(Parser Agent)]

B --> C[原始文档树<br/>(Raw Doc Tree)]

C --> D[精炼智能体<br/>(Refiner Agent)]

D --> E[内容树<br/>(Content Tree)]

E --> F[布局智能体<br/>(Layout Agent)]

F --> G[海报树<br/>(Poster Tree)]

G --> H[绘制智能体<br/>(Painter Agent)]

H --> I[渲染海报]

I --> J[多模态评论智能体<br/>(Multi-Modal Commenter)]

J --> K{全局控制智能体<br/>(Global Controller)}

K -->|评分 < 0.85| D

K -->|评分 ≥ 0.85| L[最终输出]
```

### 4.2 所需智能体（Agent）与资源配置
本实验完全基于课件推荐的免费资源实现，无需付费API。

| 智能体模块 | 职责描述 | 所需资源与工具 | 课程参考来源 |
|:---|:---|:---|:---|
| **解析智能体** | 提取文本与图表，构建文档树 | `Marker`/`Docling` (PDF解析)<br>GLM-4 (硅基流动免费API) | Paper2Poster Parser |
| **精炼智能体** | 层级化压缩，保留核心逻辑 | GPT-4o (NVIDIA/OpenRouter免费API) | PosterForest 精炼模块 |
| **语义检查智能体** | 检查跨章节逻辑一致性 | Qwen-VL (通义千问视觉模型)<br>或 GPT-4o | GenPilot VQA分支逻辑 |
| **布局智能体** | 基于权重的二叉树分区 | `python-pptx` (代码生成)<br>FreeCAD API (几何计算参考) | Paper2Poster Planner |
| **平衡评估智能体** | 评估对齐、留白、密度 | 视觉大模型 (GPT-4o/InternVL) | Paper2Poster Commenter |
| **绘制智能体** | 执行代码生成面板图像 | `python-pptx`, `ffmpeg` | Paper2Poster Painter |
| **多模态评论智能体** | 评估图文匹配度与视觉层次 | GPT-4o (NVIDIA免费API) | GenPilot MLLM评分器 |
| **全局控制智能体** | 融合评分，决定迭代终止 | Llama 3 8B (本地部署/Ollama) | PosterForest 反馈循环 |

### 4.3 核心创新点
1.  **跨节点语义约束**：新增“逻辑链”检查，确保子节点内容支撑父节点论点（如“实验结果”必须对应“方法”章节）。
2.  **动态布局权重**：不再等分画布，而是根据章节重要性（如“结论”占25%，“参考文献”占5%）动态分配面积。
3.  **自适应迭代机制**：基于综合评分动态终止，平均减少30%无效迭代。

## 5. 实验设计
### 5.1 数据集
- **主测试集**：Paper2Poster 官方测试集（50篇 NeurIPS/ICML 论文，平均22页）。
- **扩展集**：10篇跨学科论文（计算机+生物/社科），用于验证泛化性。

### 5.2 基线对比
| 基线方法 | 描述 |
|:---|:---|
| **Paper2Poster (原版)** | 固定2轮 Painter-Commenter 循环 |
| **PosterForest (原版)** | 静态树合并，无语义检查 |
| **单智能体生成** | 直接用 GPT-4o + python-pptx 生成 |

### 5.3 评估指标
| 维度 | 指标 | 测量方式 |
|:---|:---|:---|
| **内容质量** | 语义完整性 (SC) | 人工打分(1-5) + Qwen-VL 问答准确率 |
| **视觉质量** | 布局平衡分 (LB) | 视觉大模型评分（对齐/留白/图文比） |
| **效率** | 迭代次数 (IC) | 收敛所需的平均轮数 |

## 6. 实施计划
| 周次 | 时间 | 任务 | 资源 |
|:---|:---|:---|:---|
| **第1周** | 5.24-5.30 | 复现基线；构建数据集 | GitHub开源代码；NVIDIA免费API |
| **第2周** | 5.31-6.6 | 实现精炼与语义检查智能体 | LangChain框架；Ollama本地部署LLM |
| **第3周** | 6.7-6.13 | 集成布局与多模态评论模块 | GLM-4 API；python-pptx |
| **第4周** | 6.14-6.20 | 运行实验；撰写报告与PPT | LaTeX (ICLR模板)；FFmpeg |

## 7. 预期成果
1.  **量化结果**：语义完整性 >4.2/5.0，布局平衡分 >4.5/5.0，平均迭代 ≤2.1轮。
2.  **定性展示**：3个典型案例的迭代优化过程演示（GIF或视频）。
3.  **交付物**：开源代码、实验报告PDF、汇报PPT、演示视频。

## 8. 团队成员与分工
| 姓名 | 学号 | 分工 |
|:---|:---|:---|
| [姓名1] | [ID] | 系统架构设计；解析/精炼智能体实现 |
| [姓名2] | [ID] | 布局智能体；平衡评估模块开发 |
| [姓名3] | [ID] | 多模态评论；全局控制器集成 |
| [姓名4] | [ID] | 实验设计与执行；报告与PPT制作 |

## 9. 参考文献
1. Shi et al. "Paper2Poster: Towards Multimodal Poster Automation from Scientific Papers." *NeurIPS 2025*.
2. Kim et al. "PosterForest: Hierarchical Multi-Agent Collaboration for Scientific Poster Generation." *arXiv 2025*.
3. Xia et al. "GenPilot: A Multi-Agent System for Test-Time Prompt Optimization in Image Generation." *Findings of EMNLP 2025*.