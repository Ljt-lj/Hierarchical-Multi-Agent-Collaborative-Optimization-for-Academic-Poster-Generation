"""平衡评估智能体：对齐、留白、密度评估."""

from __future__ import annotations

from poster_agent.models.trees import LayoutRect, PosterNode


class BalanceAgent:
    def evaluate(self, poster_tree: PosterNode) -> tuple[float, dict[str, float]]:
        leaves = _collect_leaves(poster_tree)
        if not leaves:
            return 0.5, {"alignment": 0.5, "whitespace": 0.5, "density": 0.5}

        canvas_area = poster_tree.rect.width * poster_tree.rect.height
        used_area = sum(n.rect.area for n in leaves)
        fill_ratio = used_area / canvas_area if canvas_area else 0
        whitespace_score = 1.0 - abs(fill_ratio - 0.95) / 0.95
        whitespace_score = max(0.0, min(1.0, whitespace_score))

        align_score = _alignment_score(leaves)
        density_scores = [_density_score(n) for n in leaves]
        density_score = sum(density_scores) / len(density_scores)

        overall = 0.35 * align_score + 0.35 * whitespace_score + 0.30 * density_score
        metrics = {
            "alignment": round(align_score, 3),
            "whitespace": round(whitespace_score, 3),
            "density": round(density_score, 3),
        }
        return overall, metrics


def _collect_leaves(node: PosterNode) -> list[PosterNode]:
    if not node.children:
        if node.title not in ("Poster Root", "Section Group"):
            return [node]
        return []
    out: list[PosterNode] = []
    for c in node.children:
        out.extend(_collect_leaves(c))
    return out


def _alignment_score(leaves: list[PosterNode]) -> float:
    if len(leaves) < 2:
        return 0.9
    xs = [n.rect.x for n in leaves]
    ys = [n.rect.y for n in leaves]
    x_span = max(xs) - min(xs) if xs else 1
    y_span = max(ys) - min(ys) if ys else 1
    x_var = sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / len(xs)
    y_var = sum((y - sum(ys) / len(ys)) ** 2 for y in ys) / len(ys)
    norm = max(x_span * x_span, y_span * y_span, 1)
    variance_penalty = (x_var + y_var) / norm
    return max(0.0, min(1.0, 1.0 - variance_penalty * 2))


def _density_score(node: PosterNode) -> float:
    char_count = len(node.summary) + sum(len(b) for b in node.bullets)
    capacity = max(node.rect.area / 600, 1)
    ratio = char_count / capacity
    if 0.2 <= ratio <= 1.0:
        return 1.0
    if ratio < 0.2:
        return 0.6 + ratio * 2
    return max(0.0, 1.0 - (ratio - 1.0) * 0.5)
