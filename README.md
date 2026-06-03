# 基于层级多智能体协作的学术海报生成优化系统

计算机图形学 Project 3 实验代码。

## 系统架构

8 个智能体协作流水线（Parser → Refiner → Semantic → Layout → Balance → Painter → Commenter → Controller），评分低于 0.85 时自动回退精炼阶段迭代优化。

## 环境配置

```bash
pip install -r requirements.txt
```

API 密钥写入项目根目录 `api_key.txt`（已加入 `.gitignore`），或使用环境变量 `DEEPSEEK_API_KEY`。

DeepSeek 配置：
- Base URL: `https://api.deepseek.com`
- 默认模型: `deepseek-v4-flash`

## 快速开始

**演示模式**（无需 PDF）：

```bash
python main.py --demo
```

**指定 PDF 论文**：

```bash
python main.py path/to/paper.pdf --output-name my_poster
```

**真实论文测试**（自动从 arXiv 下载 5 篇英文论文并生成海报）：

```bash
python experiments/real_paper_test.py
python experiments/real_paper_test.py --skip-download   # 使用已下载 PDF
python experiments/real_paper_test.py --count 3
```

输出目录结构：
```
outputs/real_paper_test/{批次时间}/
  Attention_Is_All_You_Need/
    source/1706.03762.pdf
    20260602_154530_Attention_Is_All_You_Need.png
    20260602_154530_Attention_Is_All_You_Need.pptx
    20260602_154530_Attention_Is_All_You_Need_meta.json
    ...
  batch_report.json
```

**批量测试**（推荐）：

```bash
# 生成示例 PDF
python tools/generate_demo_pdfs.py

# 批量测试 PDF 目录
python experiments/batch_test.py --pdf-dir samples/papers --output outputs/batch

# 使用 manifest 清单 + 断点续跑
python experiments/batch_test.py --manifest experiments/batch_manifest.json --resume
```

**旧版批量实验**：

```bash
python experiments/run_experiment.py --demo-count 3
python experiments/run_experiment.py --pdf-dir samples/papers
```

## 可视化美化

系统自动为各章节生成可视化元素（非纯文本摘要）：
- 柱状图 / 折线图 / 饼图（实验数据）
- 流程图 / 架构图（方法章节）
- 统计卡（关键指标）
- PDF 论文插图（自动提取嵌入）

海报样式（参考学术海报范例）：
- 深蓝通栏标题 + 居中论文名
- **摘要置顶通栏**，其余章节 **双栏 masonry** 紧凑排列
- 蓝色章节标题栏 + 白底圆角内容框
- 动态字号：参考文献缩小，正文随内容密度自适应
- 图文混排模式：上图下文 / 左文右图 / 下图上文，图表占 60%+ 区块面积

## 输出

结果保存在 `outputs/` 目录：
- `poster_iterN.png` — 渲染海报图像
- `poster_iterN.pptx` — PowerPoint 格式
- `*_raw_tree.json` / `*_content_iter*.json` / `*_poster_iter*.json` — 中间树结构
- `batch_report.json` / `batch_results.csv` / `batch_report.html` — 批量测试报告
- `outputs/visuals/` — 自动生成的图表/流程图资产

## 项目结构

```
poster_agent/
  agents/          # 9 个智能体（含可视化智能体）
  models/          # 文档树/内容树/海报树/VisualSpec
  render/          # 美化渲染 + 图表生成
  config.py        # 配置
  llm_client.py    # DeepSeek API 客户端
  pipeline.py      # 主管道
main.py            # CLI 入口
experiments/       # 批量实验脚本
```

## 评估指标

| 维度 | 指标 | 说明 |
|------|------|------|
| 内容质量 | SC (语义完整性) | LLM + 逻辑链检查 |
| 视觉质量 | LB (布局平衡分) | 对齐/留白/密度 |
| 效率 | IC (迭代次数) | 自适应终止 |
