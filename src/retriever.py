"""统一检索入口：BM25 / 向量 / 混合。

这一层只做召回，不生成任何文字。返回的每条结果都挂着一个完整的 Chunk，
所以公司名、页码、章节路径全部来自 data/chunks 里的原始记录 ——
检索过程（包括向量模型）不参与文本生成，引用不会失真。

三种模式的定位：

  bm25    关键词精确匹配。数字、科目名、专有名词准。
  vector  语义近似。问法和报告原文措辞不一致时管用。
  hybrid  RRF 融合上面两路。默认。

**公司过滤**在打分前生效（BM25 用布尔掩码、向量用 -inf 掩码），
不是在拿到 top-k 之后再筛。后者的问题很实际：问"中国平安的总资产"，
top-10 里可能有 8 条中国人寿的段落，筛完只剩 2 条，等于白召回。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import config as cfg
from .bm25 import BM25Index
from .chunker import load_chunks
from .hybrid import fuse
from .schema import Chunk
from .vector_index import VectorIndex

# 兜底用；正式来源是 companies.yaml 的 aliases 字段
MODE_BM25 = "bm25"
MODE_VECTOR = "vector"
MODE_HYBRID = "hybrid"
MODES = (MODE_BM25, MODE_VECTOR, MODE_HYBRID)
MODE_ZH = {MODE_BM25: "BM25", MODE_VECTOR: "向量", MODE_HYBRID: "混合"}


@dataclass
class RetrievedChunk:
    """一条召回结果：chunk 本身 + 它怎么被召回的。"""

    chunk: Chunk
    score: float
    rank: int                      # 从 1 开始
    source: str                    # bm25 / vector / hybrid
    company_key: str = ""          # 评测用；chunk 里没存，从 doc_id 还原

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def company(self) -> str:
        return self.chunk.company

    @property
    def page(self) -> int:
        return self.chunk.page

    def citation(self, zh: bool = True) -> str:
        """作业要求的引用格式。直接代理给 Chunk，不自己拼字符串。"""
        return self.chunk.citation(zh=zh)

    def preview(self, n: int = 160) -> str:
        t = " ".join(self.chunk.text.split())
        return t[:n] + ("…" if len(t) > n else "")


def _alias_table() -> list[tuple[str, str]]:
    """[(别名, 公司key), ...]，按别名长度降序。

    必须长的排前面。否则问"人保财险的保费"，短的"人保"（如果收了）
    会先命中，判成中国人保 —— 张冠李戴。
    """
    _, profiles = cfg.load_companies()
    pairs: list[tuple[str, str]] = []
    for p in profiles:
        for a in (p.aliases or []):
            if a:
                pairs.append((a, p.key))
        # 公司全称也当别名，但不加简称 —— 简称靠 yaml 显式声明
        if p.company_full:
            pairs.append((p.company_full, p.key))
    pairs.sort(key=lambda x: -len(x[0]))
    return pairs


def _doc_key_resolver() -> list[str]:
    """所有公司 key，按长度降序。用于把 doc_id 还原成公司 key。"""
    _, profiles = cfg.load_companies()
    return sorted((p.key for p in profiles), key=len, reverse=True)


def _key_of_doc(doc_id: str, keys_by_len: list[str]) -> str:
    """doc_id（如 picc_pc_2025_annual）→ 公司 key（picc_pc）。

    必须**最长前缀**优先：`picc` 是 `picc_pc` 的前缀，按短的先匹配会把
    人保财险的所有 chunk 判成中国人保 —— 两家公司的答案会互相串。
    """
    for k in keys_by_len:
        if doc_id == k or doc_id.startswith(k + "_"):
            return k
    return doc_id.split("_")[0]


def load_corpus(chunks_dir: Path | None = None,
                *, strict: bool = True) -> tuple[list[Chunk], list[str]]:
    """加载全部 chunk，丢掉不属于配置公司的那些。

    为什么必须过滤：data/chunks 里还躺着 tests/fixtures 生成的合成样本
    （demo_2025_annual.jsonl，7 条占位内容）。它不是任何公司的年报，
    混进索引就是来源不明的数据 —— 本项目从头到尾不允许这种事。

    strict=True 时，过滤掉的会连文件名一起返回，由调用方决定是否提示。
    过滤规则和 doc_id → 公司 key 的还原规则是同一套（最长前缀），
    所以这里认不出来的，后面归属也一定认不出来。
    """
    chunks_dir = Path(chunks_dir or cfg.CHUNKS_DIR)
    keys_by_len = _doc_key_resolver()
    known = set(keys_by_len)

    kept: list[Chunk] = []
    dropped: list[str] = []
    for p in sorted(chunks_dir.glob("*.jsonl")):
        for c in load_chunks(p):
            if _key_of_doc(c.doc_id, keys_by_len) in known:
                kept.append(c)
            else:
                dropped.append(f"{p.name}:{c.chunk_id}")
    if not kept:
        raise FileNotFoundError(
            f"{chunks_dir} 下没有属于配置公司的 chunk。"
            f"先跑第一阶段：python -m src.cli batch")
    return kept, dropped


def verify_consistency(chunks: list[Chunk], indexes: dict) -> None:
    """校验索引里的 doc_ids 与 chunks 完全同序，不一致就抛错。

    这个检查必须在加载时做。索引里的 doc_idx 是个整数下标，一旦两边顺序
    或内容对不上，检索会拿着正确的分数指向**错误的 chunk** ——
    表现为"答案看着对、页码是错的"，是所有失败模式里最难发现的一种。
    """
    expect = [c.chunk_id for c in chunks]
    expected_set = set(expect)
    for name, idx in indexes.items():
        if idx is None:
            continue
        if list(idx.doc_ids) != expect:
            got = set(idx.doc_ids)
            miss = [x for x in expect if x not in got][:3]
            extra = [x for x in idx.doc_ids if x not in expected_set][:3]
            raise ValueError(
                f"{name}索引与当前 chunks 不一致（索引 {len(idx.doc_ids)} 条 / "
                f"chunks {len(expect)} 条）。缺：{miss} 多：{extra}。"
                f"chunks 变过之后必须重建索引："
                f"python scripts/build_index.py --force")


class Retriever:
    """加载 chunks + 索引，提供 search()。"""

    def __init__(self, chunks: list[Chunk],
                 bm25: BM25Index | None = None,
                 vector: VectorIndex | None = None):
        self.chunks = chunks
        self.by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
        self.bm25 = bm25
        self.vector = vector
        keys_by_len = _doc_key_resolver()
        # 每个 chunk 归属哪家公司。chunk 本身没存 key（只有中文简称），
        # 从 doc_id 还原 —— doc_id 是 "<key>_<year>_annual"。
        self.chunk_keys: list[str] = [
            _key_of_doc(c.doc_id, keys_by_len) for c in chunks]
        self.key_by_id: dict[str, str] = {c.chunk_id: k for c, k in
                                          zip(chunks, self.chunk_keys)}
        self._aliases = _alias_table()
        # 公司 -> 布尔掩码（同一家公司反复查时不用重算）
        self._mask_cache: dict[str, list[bool]] = {}

    # ------------------------------------------------------------------ 加载
    @classmethod
    def load(cls, *, chunks_dir: Path | None = None,
             bm25_dir: Path | None = None,
             vector_dir: Path | None = None,
             with_vector: bool = True,
             with_bm25: bool = True) -> "Retriever":
        chunks_dir = Path(chunks_dir or cfg.CHUNKS_DIR)
        bm25_dir = Path(bm25_dir or cfg.BM25_DIR)
        vector_dir = Path(vector_dir or cfg.VECTOR_DIR)

        chunks, dropped = load_corpus(chunks_dir)
        if dropped:
            print(f"提示：跳过 {len(dropped)} 条非配置公司的 chunk"
                  f"（如合成测试样本）：{dropped[0]}"
                  + ("…" if len(dropped) > 1 else ""))

        bm25 = BM25Index.load(bm25_dir) if with_bm25 else None
        vector = VectorIndex.load(vector_dir) if with_vector else None

        verify_consistency(chunks, {"BM25": bm25, "向量": vector})
        return cls(chunks, bm25=bm25, vector=vector)

    # ------------------------------------------------------------ 公司识别
    def detect_companies(self, query: str) -> list[str]:
        """从问题里找出提到了哪些公司。可能不止一家（"比较平安和国寿"）。

        做法是逐个别名做子串匹配，命中即算。不去重、不排序，按 key 去重后
        返回。识别不出就返回空列表 —— 空的语义是"不过滤"，而不是"没公司"。
        """
        hits: list[str] = []
        seen: set[str] = set()
        for alias, key in self._aliases:
            if key in seen:
                continue
            if alias in query:
                seen.add(key)
                hits.append(key)
        return hits

    def detect_company(self, query: str) -> str | None:
        """只识别出一家公司时返回它，其余情况返回 None（不做过滤）。

        多于一家时刻意返回 None：问题本身跨公司，硬过滤会把另一半答案切掉。
        """
        hits = self.detect_companies(query)
        return hits[0] if len(hits) == 1 else None

    def _mask_for(self, spec: str) -> list[bool]:
        """公司 key 或中文简称 → 布尔掩码。

        中文简称允许"中国平安"这种写法；重名（理论上不该有）只在
        chunk 里真的存在对应名时才采纳。
        """
        m = self._mask_cache.get(spec)
        if m is None:
            m = [k == spec or c.company == spec
                 for k, c in zip(self.chunk_keys, self.chunks)]
            if not any(m):
                known = sorted(set(self.chunk_keys))
                raise KeyError(f"没有匹配 {spec!r} 的 chunk。可用 key：{known}")
            self._mask_cache[spec] = m
        return m

    # ---------------------------------------------------------------- 检索
    def _rank_one(self, query: str, top_k: int, allowed: list[bool] | None,
                  mode: str, pool: int, weights, fusion: str
                  ) -> list[tuple[str, float]]:
        """单次召回（可能已做 BM25/向量融合），返回 [(chunk_id, score)]。"""
        rankings: dict[str, list[tuple[str, float]]] = {}
        if mode in (MODE_BM25, MODE_HYBRID):
            if self.bm25 is None:
                raise RuntimeError("BM25 索引没加载（with_bm25=False）")
            rankings["bm25"] = self.bm25.search(
                query, top_k=(top_k if mode == MODE_BM25 else pool),
                allowed=allowed)
        if mode in (MODE_VECTOR, MODE_HYBRID):
            if self.vector is None:
                raise RuntimeError("向量索引没加载（with_vector=False）")
            rankings["vector"] = self.vector.search(
                query, top_k=(top_k if mode == MODE_VECTOR else pool),
                allowed=allowed)

        if mode == MODE_BM25:
            return rankings["bm25"][:top_k]
        if mode == MODE_VECTOR:
            return rankings["vector"][:top_k]
        return fuse(rankings, method=fusion, weights=weights, top_k=top_k)

    def search(self, query: str, top_k: int = 5, *,
               company: str | None = None,
               mode: str = MODE_HYBRID,
               auto_company: bool = True,
               pool: int = 50,
               weights: dict[str, float] | None = None,
               fusion: str = "rrf",
               multi_company: str = "per_company") -> list[RetrievedChunk]:
        """召回。

        Args:
            company: 指定公司 key（如 "pingan"）或中文简称，只在该公司内检索。
            auto_company: 为 True 且未显式指定 company 时，从 query 里自动认公司。
            pool: 融合前每一路各取多少条。太小会让某一路的好结果进不了融合。
            weights: 给 {"bm25": w, "vector": w} 加权（仅 hybrid）。
            fusion: 融合方式，"rrf"（默认）/ "weighted" / "union"。
                默认值是评测选出来的：实测 rrf 与其它几种打平或更好，
                见 docs/phase2_retrieval.md 的融合配置对比。
            multi_company: 问题里提到多家公司时怎么办。
                "per_company"（默认）= 每家公司各召回 top_k 条，再按排名融合。
                "global"        = 不按公司分组，直接全库召回。

                为什么默认按公司分组：问"比较平安和国寿的净利润"，全库 top-10
                常常被正文长、重复提到该指标的那一家占满，另一家一条都进不来 ——
                实测这道题三路都只召回了一半。分组召回保证每家公司都有代表，
                用户要的是"两家分别是多少"，不是"其中一家是多少"。
        """
        if mode not in MODES:
            raise ValueError(f"未知检索模式：{mode!r}（可选 {'/'.join(MODES)}）")

        keys: list[str] = []
        if company is not None:
            keys = [company]
        elif auto_company:
            keys = self.detect_companies(query)

        if len(keys) <= 1:
            allowed = self._mask_for(keys[0]) if keys else None
            merged = self._rank_one(query, top_k, allowed, mode, pool,
                                    weights, fusion)
        elif multi_company == "global":
            merged = self._rank_one(query, top_k, None, mode, pool,
                                    weights, fusion)
        else:
            per: dict[str, list[tuple[str, float]]] = {}
            for k in keys:
                per[k] = self._rank_one(query, top_k, self._mask_for(k),
                                        mode, pool, weights, fusion)
            merged = fuse(per, method="rrf", top_k=top_k)

        out: list[RetrievedChunk] = []
        for rank, (cid, score) in enumerate(merged, start=1):
            ch = self.by_id.get(cid)
            if ch is None:              # 索引里有、chunks 里没有 —— 上面已查过，兜底
                continue
            out.append(RetrievedChunk(chunk=ch, score=float(score),
                                      rank=rank, source=mode,
                                      company_key=self.key_by_id.get(cid, "")))
        return out

    def search_per_source(self, query: str, top_k: int = 5, *,
                          company: str | None = None,
                          auto_company: bool = True,
                          pool: int = 50,
                          fusion: str = "rrf",
                          weights: dict[str, float] | None = None
                          ) -> dict[str, list[RetrievedChunk]]:
        """一次拿到三路结果，供评测逐路对比。"""
        return {
            MODE_BM25: self.search(query, top_k, company=company, mode=MODE_BM25,
                                   auto_company=auto_company),
            MODE_VECTOR: self.search(query, top_k, company=company, mode=MODE_VECTOR,
                                     auto_company=auto_company),
            MODE_HYBRID: self.search(query, top_k, company=company, mode=MODE_HYBRID,
                                     auto_company=auto_company, pool=pool,
                                     fusion=fusion, weights=weights),
        }

    # ---------------------------------------------------------------- 统计
    def stats(self) -> dict:
        by_key: dict[str, int] = {}
        pages: dict[str, set] = {}
        for k, c in zip(self.chunk_keys, self.chunks):
            by_key[k] = by_key.get(k, 0) + 1
            pages.setdefault(k, set()).add(c.page)
        return {
            "chunks": len(self.chunks),
            "companies": len(by_key),
            "by_company": dict(sorted(by_key.items(), key=lambda x: -x[1])),
            "pages_covered": {k: len(v) for k, v in sorted(pages.items())},
            "bm25": self.bm25.stats() if self.bm25 else None,
            "vector": self.vector.stats() if self.vector else None,
        }


def format_results(results: list[RetrievedChunk], *, show_text: int = 0) -> str:
    """把召回结果打印成人能读的样子。"""
    lines = []
    for r in results:
        lines.append(f"[{r.rank}] {r.score:.4f}  {r.chunk_id}")
        lines.append(f"    {r.citation()}")
        if show_text:
            lines.append(f"    {r.preview(show_text)}")
    return "\n".join(lines)


__all__ = ["Retriever", "RetrievedChunk", "format_results", "load_corpus",
           "verify_consistency", "MODE_BM25", "MODE_VECTOR", "MODE_HYBRID",
           "MODES", "MODE_ZH"]
