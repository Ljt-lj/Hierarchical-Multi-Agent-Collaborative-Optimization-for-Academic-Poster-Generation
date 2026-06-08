# PosterAgent: Training-Free Hierarchical Multi-Agent Optimization for Academic Poster Generation

> **Course Report Script (English)** — Computer Graphics, Spring 2026, Project 3  
> **Recommended typesetting:** Convert this Markdown to LaTeX using the [ICLR Master Template](https://github.com/ICLR/Master-Template/).  
> **Submission format:** PDF

---

## Abstract

Academic posters remain a primary medium for communicating research at conferences, yet manual design is time-consuming and existing automation tools often produce posters with fragmented logic, poor figure readability, and rigid iteration schedules. We present **PosterAgent**, a **training-free hierarchical multi-agent framework** that transforms PDF research papers into publication-quality academic posters without fine-tuning any layout or vision model. Our system introduces three key innovations: (1) a **logic-first content pipeline** that extracts a cross-section *LogicPlan* and enforces semantic dependencies between Methods, Results, and Discussion; (2) a **semantically assigned three-column academic layout** (25% / 50% / 25%) coupled with **content-level column balancing** that adjusts text rather than distorting typography; and (3) a **paper-native figure pipeline** that renders, role-crops, and inserts PDF figures at panel boundaries to avoid mid-panel clipping and unreadably small visuals. A **multi-metric adaptive controller** fuses semantic completeness, layout balance, and image–text match scores to terminate refinement early when quality thresholds are met. On a challenging Nature Methods paper (*GRASP*, protein complex structure prediction), our system improves the composite quality score from **0.53 to 0.86** within three iterations and produces a final poster with intelligently cropped figures and ORF3a-style panel structure. Code and experiment artifacts are publicly available at: **[https://github.com/YOUR_USERNAME/poster-agent](https://github.com/YOUR_USERNAME/poster-agent)** *(replace with your repository URL before submission)*.

**Keywords:** academic poster generation, multi-agent systems, document layout, figure extraction, LLM orchestration, training-free optimization

---

## 1. Introduction

Conference posters condense long-form papers into a single visual narrative. Unlike slides or short summaries, posters must simultaneously satisfy **information completeness**, **logical coherence**, **visual hierarchy**, and **spatial efficiency** within a fixed canvas (typically 36×24 inches or equivalent pixel dimensions). Researchers spend hours manually selecting figures, writing bullet points, and balancing columns—tasks that are repetitive yet highly structured.

Recent work has explored automating poster generation with large language models (LLMs). **Paper2Poster** decomposes the task into Parser–Planner–Painter stages; **PosterForest** organizes content as a hierarchical tree; **GenPilot** applies error analysis to refine text-to-image pipelines. However, these approaches share critical limitations:

1. **Logic fragmentation.** Section summaries are generated independently. A Results bullet may cite a benchmark never introduced in Methods, breaking the reader's causal chain.
2. **Figure misuse.** Embedded PDF images are often low-resolution fragments. Naive scaling either crops figures at arbitrary boundaries ("waist-cutting") or shrinks multi-panel figures until axis labels become illegible.
3. **Single-dimensional feedback.** Overflow detection alone cannot judge whether figures support claims or whether columns are narratively balanced.
4. **Fixed iteration budgets.** Two-pass painter loops waste compute on easy papers and under-optimize hard ones.

We address these gaps with **PosterAgent**, implemented as a compositional pipeline of **15+ specialized agents** orchestrated over three explicit data structures: a **Raw Document Tree** (parsed PDF), a **Content Tree** (refined semantic blocks), and a **Poster Tree** (spatial layout with bounding rectangles). The system requires only API access to an instruction-following LLM (DeepSeek-V4-Flash in our deployment) and standard Python libraries (PyMuPDF, Pillow, Matplotlib, python-pptx)—**no model training**.

Our contributions are:

- **C1. Logic-first multi-agent content generation.** A *LogicPlannerAgent* extracts pipeline steps and validation chains before refinement. *RefinerAgent* attaches `logic_links` across sections; *SemanticCheckAgent* scores cross-section consistency.
- **C2. Semantically assigned three-column academic layout.** Inspired by professional biology posters (e.g., ORF3a reference layouts), we assign Introduction/Methods to the left column, Results to the wide center column (50%), and Conclusions/Future Work/References to the right column—implemented as title-driven routing, not generic grid packing.
- **C3. Section expander and content-level column balancer.** Flat LLM sections are expanded into rich panel stacks (e.g., four Results sub-panels). A *column_balancer* estimates column height from layout plans and expands/trims bullet content to equalize loads **without line-spacing hacks**.
- **C4. Paper-native figure pipeline with role-aware cropping.** We render full figure pages from PDF, detect horizontal whitespace bands to split multi-panel figures, and produce role-specific crops (`intro`, `benchmark`, `structure`) assigned to poster panels—maximizing readable figure area while preserving panel integrity.
- **C5. Multi-metric adaptive control with dual feedback loops.** An outer loop refines content via *ControllerAgent* (weighted SC + LB + ITM); an inner loop re-layouts sections with `height_boost` when *PosterRenderer* reports overflow.

We validate PosterAgent on **GRASP** (Zhang et al., *Nature Methods* 2025), a 22-page paper with complex multi-panel figures, benchmark tables (DockQ), and method pipeline diagrams. The final poster (`s41592_02820_v19_result.png`) demonstrates all innovations working in concert.

---

## 2. Related Work

### 2.1 Automated Poster and Slide Generation

**Paper2Poster** (2024) pioneered a Parser–Planner–Painter decomposition for converting papers to posters. Its planner assigns regions on a canvas and its painter renders panels. However, it lacks explicit cross-section logic verification and uses relatively simple overflow feedback. **PosterForest** (2024) treats poster content as a mergeable hierarchical forest, enabling joint optimization of text blocks. Its iteration count is fixed, and figure handling remains decoupled from section semantics.

Commercial tools (Canva, PowerPoint Designer) offer templates but no paper-aware parsing. Single-prompt LLM approaches ("Summarize this PDF into poster bullets") ignore layout geometry and figure provenance entirely.

### 2.2 Document Understanding and Figure Extraction

PDF parsing tools such as **GROBID**, **Marker**, and **Docling** extract structured text but do not reason about poster narrative. Our *ParserAgent* uses PyMuPDF with custom heuristics to detect figure caption pages, render focus panels (`figure_page_renderer.py`), and build a scored *FigureCatalog* (`figure_curator.py`). Unlike naive embedded-image extraction, we render **page regions** at 220–240 DPI, preserving vector-quality figures.

### 2.3 LLM Multi-Agent Orchestration

**AutoGen**, **LangGraph**, and **CrewAI** provide generic agent frameworks. PosterAgent instead embeds **domain-specific agents** with typed interfaces (trees, visual specs, render reports). This mirrors GenPilot's insight that **error analysis should route to specialized fixers**—our *PosterErrorAnalyzer* maps overflow/sparsity to refiner, layout, or visual actions.

### 2.4 Layout and Visualization

Automatic document layout has a long history in computational geometry and GUI toolkits. Academic posters differ: sections carry **semantic roles** (Results must be visually dominant). We combine weighted section heights (`section_weights` in config) with masonry packing (dual-column mode) and a dedicated **three-column academic mode** (`GridLayoutEngine._layout_three_column`).

### 2.5 Positioning Summary

| Method | Logic chain | Figure pipeline | Column semantics | Adaptive iteration |
|--------|-------------|-----------------|------------------|--------------------|
| Paper2Poster | Partial | Embedded images | Generic regions | Fixed 2-pass |
| PosterForest | Tree merge | Limited | Static weights | Fixed |
| Single LLM + PPTX | None | Manual/none | Template | Single shot |
| **PosterAgent (Ours)** | **LogicPlan + checks** | **Render + role crop** | **Title-routed 3-col** | **Score threshold** |

---

## 3. Method

### 3.1 System Overview

PosterAgent executes the following pipeline (`poster_agent/pipeline.py`):

```
PDF → ParserAgent → LogicPlannerAgent
     → [LOOP until score ≥ τ or max iterations]
         RefinerAgent → SectionExpander* → VisualAgent
         → ColumnBalancer* → PainterAgent.prepare_visuals
         → SemanticCheckAgent → LayoutAgent → PainterAgent.paint
         → PosterErrorAnalyzer → BalanceAgent → CommenterAgent
         → ControllerAgent
     → Best-iteration PNG/PPTX
```

\* *SectionExpander* and *ColumnBalancer* activate when `three_column_layout=True` and `academic_style=True` (defaults).

**Three-tree representation** (`poster_agent/models/trees.py`):

| Tree | Purpose | Key fields |
|------|---------|------------|
| `RawNode` | Parsed document | `content`, `children`, `tables`, `figure_catalog` |
| `ContentNode` | Refined poster content | `bullets`, `logic_links`, `visuals`, `paper_tables`, `block_style` |
| `PosterNode` | Spatial layout | `rect`, `layout_mode`, `image_paths`, `font_size_*` |

This separation allows **targeted feedback**: semantic issues route to *RefinerAgent*; overflow routes to *LayoutAgent* with `height_boost`; missing visuals route to *VisualAgent*.

### 3.2 Parsing and Logic Planning

**ParserAgent** (`parser_agent.py`):
- Extracts hierarchical sections from PDF text.
- Builds `figure_catalog` via `render_figure_focus_panels()` and `render_figure_composites()` (`figure_page_renderer.py`).
- Extracts benchmark tables (DockQ, success rates) via `table_extractor.py`.

**LogicPlannerAgent** (`logic_planner_agent.py`) runs **before** refinement and outputs a `LogicPlan`:

```json
{
  "core_problem": "AFM/AF3 limited accuracy; integrate sparse XL-MS/NMR restraints",
  "pipeline_steps": ["Input sequences + RPR/IR graph", "RPR → MSA/IPA bias", ...],
  "validation_chain": ["DockQ benchmark", "Ablation on restraint count", ...],
  "key_terms": ["RPR", "IR", "AFM", "GRASP", "DockQ"]
}
```

The refiner must preserve `logic_links` such as `Methods→Results: benchmark validation`, enabling downstream semantic checks.

### 3.3 Content Refinement and Section Expansion

**RefinerAgent** compresses each section into poster-appropriate bullets while retaining metrics (e.g., "GRASP achieves mean DockQ 0.87 vs AF3 0.17").

**Section expander** (`section_expander.py`) transforms flat sections into **ORF3a-style panel stacks**:

| Original section | Expanded structure |
|------------------|-------------------|
| Results | (1) Benchmark across restraint types [Fig.2 crop] (2) Structure prediction [Fig.3 crop] (3) DockQ bar chart (4) Key quantitative findings |
| Methods | (1) Graph construction & encoding (2) GRASP integration pipeline [logic_pipeline visual] |
| Discussion | Conclusions + Future Directions + auto Acknowledgements + References |

Panels use `block_style="panel"`; parents use `layout_mode="panel_stack"` in the renderer.

### 3.4 Visual Enrichment and Deduplication

**VisualAgent** (`visual_agent.py`) plans `VisualSpec` objects per section:

- `logic_pipeline` for Methods (paper-specific steps, not generic "Input/Process/Output")
- `bar_chart` for DockQ benchmark (data from `paper_tables`)
- Skips synthetic charts when `prefer_paper_figures=True` and paper figures exist

**VisualRegistry** (`visual_registry.py`) prevents duplicate numeric series, flow steps, and stat cards across sections.

**VisualEvaluator** (`visual_evaluator.py`) runs up to 3 generation attempts per visual, with rule-based and LLM scoring (pass threshold 0.78). DockQ bar charts with validated table data are **locked** to prevent LLM refinement from collapsing multi-bar charts to a single bar.

### 3.5 Paper-Native Figure Pipeline (Core Innovation)

Naive poster tools assign embedded 150×150 px thumbnails. PosterAgent implements a four-stage figure pipeline:

**Stage 1 — Page rendering.** `render_figure_focus_panels()` clips PDF pages using caption-aware vertical bounds (`POSTER_FOCUS_CROPS`) at 240 DPI.

**Stage 2 — Role-aware cropping.** `figure_cropper.py`:
- `split_horizontal_bands()` detects whitespace rows (≥96.5% white pixels, gap ≥14 px).
- `_pick_band_for_role()` selects bands by figure number and poster role:
  - Fig.1 → `intro`: panel (a) workflow only
  - Fig.2 → `benchmark`: plot panels without bottom structure row
  - Fig.3 → `structure`: structure comparison row (panel b)
- Outputs: `fig{N}_crop_{intro|benchmark|structure}.png`

**Stage 3 — Catalog and assignment.** `figure_curator.normalize_poster_figure_paths()` maps crops to panels by title semantics (`_is_paper_benchmark_panel`, `_is_structure_panel`).

**Stage 4 — Display.** `prepare_paper_figure()` trims near-white margins; `PosterRenderer._paste_figures()` uses **contain mode** (never cover-crop) and allocates up to **42% of panel height** to paper figures.

This pipeline directly addresses **waist-cutting** (crop at panel gaps, not mid-chart) and **illegible scaling** (show fewer panels larger rather than entire figures tiny).

### 3.6 Layout Engine

**GridLayoutEngine** (`grid_layout.py`) supports two modes:

1. **Masonry dual-column** (legacy conference style): abstract full-width + weighted masonry.
2. **Three-column academic** (default): 3600×2400 px, `column_fracs=(0.25, 0.50, 0.25)`.

Column assignment (`_academic_column(title)`):

| Column | Sections |
|--------|----------|
| Left (25%) | Introduction, Methods |
| Center (50%) | Results, benchmarks, performance panels |
| Right (25%) | Conclusions, Future Directions, Acknowledgements, References |

**Height allocation:**
- `_allocate_column_stack()` stacks sections by estimated content height; slack goes to the **last section in each column** (not evenly distributed, avoiding inter-section voids).
- `_layout_panel_stack()` divides Results/Methods parent height among child panels with role-specific fractions (Results: 28/28/24/20%).

**Column balancer** (`column_balancer.py`):
- Estimates each column's load using `_plan_section_with_panels()`.
- Expands References, Acknowledgements, Future Work bullets when a column is short.
- Trims excess bullets from overflowing Results panels.
- Operates on **content**, not CSS—preserving typography integrity.

### 3.7 Rendering and Inner Feedback Loop

**PosterRenderer** (`poster_renderer.py`):
- Academic theme: white background, orange section headers (`#C45C26`), bold panel titles above figures.
- Tracks per-section overflow via `_FitTracker` → `PosterRenderReport`.
- Supports layout modes: `text_only`, `figure_top`, `figure_panel`, `panel_stack`, `side_by_side`.

**Inner loop:** If overflow sections exist, *LayoutAgent* re-runs with `height_boost[section_title] > 1.0` for up to `inner_paint_passes` attempts.

### 3.8 Multi-Metric Evaluation and Adaptive Control

**Metrics** (`EvaluationScore` in `trees.py`):

$$\text{semantic\_completeness} = 0.6 \cdot SC + 0.4 \cdot \text{logic\_score}$$

$$\text{overall} = 0.4 \cdot \text{semantic\_completeness} + 0.35 \cdot LB + 0.25 \cdot ITM$$

| Symbol | Agent | Description |
|--------|-------|-------------|
| SC | CommenterAgent | LLM semantic completeness (1–5 normalized) |
| logic_score | SemanticCheckAgent | Workflow/validation chain consistency |
| LB | BalanceAgent | Alignment, whitespace fill (~95% target), density, overflow penalty |
| ITM | CommenterAgent | Image–text match and hierarchy clarity |

**ControllerAgent** stops when `overall ≥ score_threshold` (default **0.85**) or `iteration ≥ max_iterations` (default **5**). The pipeline checkpoints the **best-scoring iteration**, not the last.

**PosterErrorAnalyzer** (GenPilot-inspired) converts render diagnostics into refiner instructions: `"Results panel overflow → trim bullets"` or `"Discussion missing visual → add bullet_cards"`.

---

## 4. Experimental Results

### 4.1 Experimental Setup

**Platform.** Python 3.10+, PyMuPDF, Pillow, Matplotlib, python-pptx; LLM: DeepSeek-V4-Flash via API.

**Primary case study.** *Integrating diverse experimental information to assist protein complex structure prediction by GRASP* (Zhang et al., *Nat. Methods* 2025), PDF: `s41592-025-02820-1.pdf`, 22 pages, multi-panel figures, DockQ benchmark tables.

**Canvas.** 3600×2400 px landscape (`PosterConfig.width/height`), three-column academic mode, abstract merged into Introduction.

**Baselines (conceptual comparison).**

| Baseline | Description |
|----------|-------------|
| B1: Single-shot LLM | One prompt → bullets → fixed template PPTX |
| B2: Paper2Poster-style | Parser + 2 fixed painter passes, no logic check |
| B3: PosterAgent w/o crop | Full focus figures, no `figure_cropper` |
| B4: PosterAgent w/o column balancer | Three-column layout without content balancing |
| **B5: PosterAgent (Full)** | All modules enabled |

**Metrics.** Composite overall score (§3.8), iteration count (IC), qualitative figure readability (manual 1–5 scale on axis label legibility and panel integrity).

### 4.2 Iterative Optimization on GRASP (v16)

We ran the full adaptive pipeline (`output_name=s41592_02820_v16`). Score history (`s41592_02820_v16_scores.json`):

| Iteration | logic_score | LB | ITM | SC (blend) | **Overall** | Key issue |
|-----------|-------------|-----|-----|------------|-------------|-----------|
| 1 | 0.65 | 0.786 | 0.20 | 0.50 | **0.525** | Severe overflow; empty section titles |
| 2 | 0.60 | 0.825 | 0.60 | 0.54 | **0.655** | Sparse Results; truncated summaries |
| 3 | 0.90 | 0.814 | 0.95 | 0.84 | **0.858** | Minor Methods/Results clip; strong ITM |

The controller converged at iteration 3 (above τ=0.85). **ITM improved from 0.20 → 0.95**, confirming that figure assignment and visual QA were primary bottlenecks early on. Logic score reached **0.90** after `logic_links` stabilized across Methods→Results.

### 4.3 Ablation: Layout and Figure Pipeline (v8–v19)

We conducted **12+ engineering iterations** on the same GRASP paper, documented in `outputs/Integrating_diverse_experimental_information_to_assist_prote/`. Qualitative progression:

| Version | Focus | Observed issue |
|---------|-------|----------------|
| v8–v10 | Dual-column masonry | Excessive whitespace; Results under-weighted |
| v14–v15 | Academic style switch | Orange headers; abstract placement |
| v16 | Three-column + balancer | Score 0.858; figure panels still dense |
| v17 | Line-spacing fill (reverted) | Distorted typography |
| v18–v19 | ORF3a panel stack | `panel_stack` layout bug (panels at origin) |
| v19_fix | Painter attach fix | Restored 3-column structure |
| **v19_crop (Final)** | **Role-aware figure crop** | **Readable figures; no waist-cutting** |

**Final result:** `s41592_02820_v19_result.png` (copied from `s41592_02820_v19_crop.png`).

Qualitative improvements in the final poster:

1. **Introduction:** `fig1_crop_intro.png` — GRASP scheme panel (a) only, ~2× larger than full Fig.1.
2. **Results benchmark panel:** `fig2_crop_benchmark.png` — box/violin plots without bottom structure row; axis labels readable.
3. **Results structure panel:** `fig3_crop_structure.png` — four-method structure comparison row (AFM/HADDOCK/AlphaLink/GRASP).
4. **DockQ panel:** Synthesized bar chart (0.87 / 0.02 / 0.17) locked against evaluator collapse.
5. **Right column:** Conclusions, Future Directions, Acknowledgements, 6 References — no large inter-section gaps (slack packing fix).

### 4.4 Figure Crop Ablation

| Configuration | Fig.2 readable? | Fig.3 readable? | Mid-panel clip? |
|---------------|-----------------|-----------------|-----------------|
| Full `fig2_focus.png` | No (structure row shrinks plots) | No (4 stacked panels) | No |
| `fig2_crop_benchmark` | **Yes** | — | **No** |
| Full `fig3_focus.png` | — | No | No |
| `fig3_crop_structure` | — | **Yes** | **No** |

Whitespace-band splitting correctly identified horizontal gaps between GRASP figure panels (validated on `s41592-025-02820-1_images/`).

### 4.5 Batch and Cross-Paper Generalization

We additionally built batch infrastructure (`experiments/batch_test.py`, `experiments/real_paper_test.py`) for multi-paper evaluation on demo PDFs and arXiv downloads (Attention Is All You Need, etc.). Batch reports export `batch_results.csv` and HTML summaries with per-paper scores. Full 60-paper evaluation per `project.md` remains future work; the GRASP case study serves as a **stress test** (complex biology figures + numeric tables + long methods).

### 4.6 Efficiency

On GRASP, adaptive termination completed in **3 iterations** vs a fixed 5-pass budget (40% reduction). Average LLM calls per iteration: ~8 agents × 1–3 attempts for visuals. Re-render from saved JSON (`tools/rerender_poster.py`) completes in **~100 s** without re-parsing PDF.

### 4.7 Limitations

- **LLM dependency:** Quality varies with model capability; we use DeepSeek-V4-Flash for cost efficiency.
- **Figure heuristics:** Whitespace-band splitting assumes horizontally separated panels; vertically stacked subfigures may need manual crop profiles.
- **Language:** Primary development on English papers; Chinese section titles supported via `language.py` but less tested.
- **Evaluation:** Composite score relies partly on LLM self-judgment; human expert study pending.

---

## 5. Conclusion

We presented **PosterAgent**, a training-free hierarchical multi-agent system for academic poster generation from PDF papers. By combining logic-first content refinement, semantically routed three-column layout, content-level column balancing, and a paper-native figure pipeline with role-aware cropping, our approach produces posters that are **logically coherent**, **visually balanced**, and **figure-readable**—addressing failure modes of naive LLM-and-template baselines.

On the GRASP *Nature Methods* paper, composite quality improved from **0.53 to 0.86** over three adaptive iterations, and the final cropped poster demonstrates professional ORF3a-style panel structure with intelligently selected figure regions.

Future work includes: (1) human evaluation study with domain experts; (2) learned figure-panel segmentation as an optional module; (3) interactive user editing of panel crops; (4) full-scale benchmark on 60 Paper2Poster papers.

---

## References

1. Zhang, C., et al. Integrating diverse experimental information to assist protein complex structure prediction by GRASP. *Nature Methods* (2025). doi:10.1038/s41592-025-02820-1
2. Paper2Poster authors. Paper2Poster: Towards Multimodal Poster Automation from Scientific Papers. arXiv (2024).
3. PosterForest authors. PosterForest: Hierarchical Tree Optimization for Poster Generation. (2024).
4. GenPilot authors. GenPilot: Generalized Visual Agent with Error Analysis. (2024).
5. Jumper, J., et al. Highly accurate protein structure prediction with AlphaFold. *Nature* 596, 583–589 (2021).
6. Abramson, J., et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* 630, 493–500 (2024).
7. OpenReview. ICLR 2026 Author Guidelines. https://iclr.cc/Conferences/2026/AuthorGuide
8. PyMuPDF Documentation. https://pymupdf.readthedocs.io/
9. DeepSeek. DeepSeek-V4 Model Card. https://api.deepseek.com/

---

## Appendix A: LaTeX Conversion Notes

To convert this script to ICLR LaTeX:

1. Clone https://github.com/ICLR/Master-Template/
2. Map sections: `\section{Introduction}`, `\section{Related Work}`, etc.
3. Replace markdown tables with `booktabs` `tabular` environments.
4. Include final poster as `\includegraphics[width=\linewidth]{figures/s41592_02820_v19_result.png}`
5. Add `\url{https://github.com/YOUR_USERNAME/poster-agent}` in Abstract (required by assignment)

## Appendix B: Key Code Entry Points

| Component | Path |
|-----------|------|
| CLI | `main.py` |
| Pipeline | `poster_agent/pipeline.py` |
| Config | `poster_agent/config.py` |
| Section expander | `poster_agent/agents/section_expander.py` |
| Column balancer | `poster_agent/agents/column_balancer.py` |
| Figure cropper | `poster_agent/extract/figure_cropper.py` |
| Grid layout | `poster_agent/render/grid_layout.py` |
| Renderer | `poster_agent/render/poster_renderer.py` |
| Re-render tool | `tools/rerender_poster.py` |
| Final poster | `outputs/.../s41592_02820_v19_result.png` |

## Appendix C: Group Members and Division of Labor

> **Required by course instructions.** Fill in before PDF submission.

| Name | Student ID | Division of Labor |
|------|------------|-------------------|
| [Member 1] | [ID] | System architecture; Parser/Refiner/LogicPlanner agents; pipeline integration |
| [Member 2] | [ID] | Layout engine (`grid_layout.py`); three-column academic mode; column balancer |
| [Member 3] | [ID] | Figure pipeline (`figure_page_renderer`, `figure_cropper`, `figure_curator`); Painter/Renderer |
| [Member 4] | [ID] | Evaluation agents (Commenter, Balance, Controller); experiments; report & demo video |

**Code repository:** [https://github.com/YOUR_USERNAME/poster-agent](https://github.com/YOUR_USERNAME/poster-agent)  
**Dataset / paper PDF:** `s41592-025-02820-1.pdf` (GRASP, *Nat. Methods* 2025, publicly available via publisher)  
**Generated poster artifact:** `outputs/Integrating_diverse_experimental_information_to_assist_prote/s41592_02820_v19_result.png`

---

*End of report script.*
