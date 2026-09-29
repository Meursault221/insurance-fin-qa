"""BM25 与向量召回的融合。

两路召回的分数量纲完全不同：BM25 是无上界的正数，余弦在 [-1, 1]。
直接加权相加必须先把两路分数校准到同一尺度，而校准方法（min-max？
z-score？按 top-1 归一？）没有一个是稳的 —— 换一个查询，分数分布就变了。

所以默认用 **RRF（Reciprocal Rank Fusion）**：

    score(d) = Σ_r  weight_r / (k + rank_r(d))

只看排名，不看原始分数，因而对量纲免疫。k 取 60 是 RRF 原论文的经验值：
k 越大，靠前的名次之间差别越小（融合越"平"）。

也提供 weighted 模式（归一化分数加权），但它是对分数敏感的，
只有在 RRF 明显更差时才值得用 —— 评测脚本会把两种模式都跑出来对比。
"""

from __future__ import annotations

from collections import defaultdict

RRF_K = 60


def rrf_fuse(rankings: dict[str, list[tuple[str, float]]],
             weights: dict[str, float] | None = None,
             k: int = RRF_K,
             top_k: int | None = None) -> list[tuple[str, float]]:
    """RRF 融合多路召回。

    Args:
        rankings: {路名: [(chunk_id, score), ...]}，每路按分数降序
        weights:  各路权重，缺省为 1.0
        k:        RRF 平滑常数
        top_k:    返回条数
    """
    weights = weights or {}
    fused: dict[str, float] = defaultdict(float)
    for name, ranked in rankings.items():
        w = weights.get(name, 1.0)
        if w == 0:
            continue
        for rank, (cid, _score) in enumerate(ranked, start=1):
            fused[cid] += w / (k + rank)
    out = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
    return out[:top_k] if top_k else out


def _minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:                  # 全部同分，全给 1，避免除零
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def weighted_fuse(rankings: dict[str, list[tuple[str, float]]],
                  weights: dict[str, float] | None = None,
                  top_k: int | None = None) -> list[tuple[str, float]]:
    """归一化分数加权融合。每条 chunk 取各路分数的加权和。

    只在某一路里出现的 chunk，另一路记为 0 分 —— 这是它比 RRF 脆的地方：
    名次再高，只要另一路完全没召回，也会被拉低。对照用。
    """
    weights = weights or {}
    acc: dict[str, float] = defaultdict(float)
    seen: set[str] = set()
    for name, ranked in rankings.items():
        w = weights.get(name, 1.0)
        if w == 0:
            continue
        normed = _minmax([s for _c, s in ranked])
        for (cid, _s), ns in zip(ranked, normed):
            acc[cid] += w * ns
            seen.add(cid)
    out = sorted(acc.items(), key=lambda x: (-x[1], x[0]))
    return out[:top_k] if top_k else out


def union_fuse(rankings: dict[str, list[tuple[str, float]]],
               weights: dict[str, float] | None = None,
               k: int = RRF_K,
               top_k: int = 10,
               reserve: int | None = None) -> list[tuple[str, float]]:
    """候选集限定为「各路前 reserve 名的并集」，并集内仍按 RRF 排序。

    和 rrf_fuse 的区别在于**候选从哪来**。rrf_fuse 把各路完整 pool 都丢进
    融合；当 pool 远大于 top_k（比如 pool=50、top_k=10）时，向量那一路排在
    11~50 名的候选虽然 RRF 分不高，但数量多，能把 BM25 本来名次靠后的正确
    答案挤出 top-10。

    这个机制是真实的，但它针对的是**症状**。当初做它是为了救"BM25 单独能
    召回、混合反而召不回"的两道题（中国太保/中国太平的总资产），实测没救回来
    —— 而且真正的原因根本不在融合层：是 BM25 的 b=0.75 把承载答案的长 chunk
    罚得太狠，让它们在**BM25 自己的排名里**就掉到了第 8、第 9。把 b 调到 0.1
    之后这两条升到第 1、第 4，融合再怎么样也丢不掉，见 src/bm25.py。

    所以默认**不用**它。留着是为了让 docs/phase2_retrieval.md 里那组对比
    可复现，不是为了用 —— 现在的成绩是 26/29，和默认的 rrf 打平。
    """
    reserve = reserve if reserve is not None else top_k
    fused = dict(rrf_fuse(rankings, weights, k=k))
    order = [cid for cid, _ in sorted(fused.items(), key=lambda x: (-x[1], x[0]))]

    candidates: set[str] = set()
    for ranked in rankings.values():
        candidates.update(cid for cid, _ in ranked[:reserve])

    kept = [cid for cid in order if cid in candidates]
    return [(cid, fused[cid]) for cid in kept[:top_k]]


def fuse(rankings: dict[str, list[tuple[str, float]]],
         method: str = "rrf",
         weights: dict[str, float] | None = None,
         top_k: int | None = None,
         reserve: int | None = None) -> list[tuple[str, float]]:
    if method == "rrf":
        return rrf_fuse(rankings, weights, top_k=top_k)
    if method == "weighted":
        return weighted_fuse(rankings, weights, top_k=top_k)
    if method == "union":
        if top_k is None:
            top_k = 10
        return union_fuse(rankings, weights, top_k=top_k, reserve=reserve)
    raise ValueError(
        f"未知融合方式：{method!r}（可选 rrf / weighted / union）")
