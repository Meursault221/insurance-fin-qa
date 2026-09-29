"""BM25 关键词检索。

实现 BM25-Okapi（k1=1.2, b=0.1）：

    idf(t)  = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score   = Σ_t idf(t) · tf·(k1+1) / (tf + k1·(1 - b + b·|d|/avgdl))

**b 明显低于常见默认值 0.75，这是在本项目的语料上实测出来的，不是抄错。**
b 控制长度归一化的强度：b=1 时长文档被罚得最狠，b=0 完全不罚。教科书上的
0.75 是在网页/新闻语料上调的 —— 那种语料里，文档越长越"泛"，被罚是合理的。
但财报 chunk 正相反：**承载答案的往往是长段落或长表格**（一页合并资产负债表
能切出几十个数字），而短 chunk 多是目录、页眉、单行小标题。按 0.75 罚长度，
等于专门压制最有信息量的那些 chunk。

实测（29 道题、top-10，见 docs/phase2_retrieval.md）：

    b=0.75（旧默认）  混合 24/29   目标召回 83.9%   MRR 0.519
    b=0.1            混合 26/29   目标召回 90.3%   MRR 0.633

领先是稳定的：k1 取 1.2 / 1.5 / 2.0 三种值时 b≈0.1 都是最好的一档；
把 29 道题对半分成两折、在一折上选 b、到另一折上验，b≈0.1 在**两个方向**
都赢，b=0.75 在两个方向都是最差或并列最差。所以这不是在这批题上调出来的假象。

另外两个实现上的选择：

1. 不用 `rank_bm25` 库，是因为需要**按公司过滤**。用户问"中国平安的总资产"，
   不该召回中国人寿的段落。过滤必须在打分前做（而不是取回 top-k 之后再筛），
   否则 top-k 会被其他公司的段落占满，筛完只剩一条。这里用布尔掩码实现。
2. 分词器是自己写的（见 src/tokenize.py），要保证建索引和检索用的是同一套，
   否则 tf/idf 统计对不上。

索引文件用 pickle 落盘：几十万词条的嵌套 dict 用 JSON 存又大又慢。
索引是本地生成的中间产物（index/ 已 gitignore），不存在来源问题。
"""

from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import tokenize as tk

# 本项目的取值，理由见上面模块开头的说明。改这里必须重建索引 ——
# 不重建也没关系，load() 会用这里的值覆盖掉索引文件里存的旧值。
K1 = 1.2
B = 0.1

INDEX_NAME = "bm25.pkl"


@dataclass
class BM25Index:
    """倒排索引 + 打分所需的统计量。"""

    doc_ids: list[str] = field(default_factory=list)
    # term -> [(doc_idx, tf), ...]
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)
    doc_len: list[int] = field(default_factory=list)
    avgdl: float = 0.0
    k1: float = K1
    b: float = B
    bigrams: bool = True

    # ---------------- 建索引 ----------------
    @classmethod
    def build(cls, chunks, *, bigrams: bool = True,
              text_of=lambda c: c.text) -> "BM25Index":
        """从 chunk 列表建索引。

        Args:
            chunks: Chunk 列表
            bigrams: 是否启用中文 bigram
            text_of: 取哪段文本建索引（默认 chunk 正文，不含 context_prefix）
        """
        idx = cls(bigrams=bigrams)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        df: Counter = Counter()

        for i, ch in enumerate(chunks):
            idx.doc_ids.append(ch.chunk_id)
            toks = tk.tokenize(text_of(ch), bigrams=bigrams)
            idx.doc_len.append(len(toks))
            tf = Counter(toks)
            for term, n in tf.items():
                postings[term].append((i, n))
            for term in tf:
                df[term] += 1

        n = len(idx.doc_ids)
        idx.avgdl = (sum(idx.doc_len) / n) if n else 0.0
        idx.postings = dict(postings)
        # 加 1 平滑：df=N 的词 idf 会变成 0 而不是负数，避免"出现越多反而扣分"
        idx.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5))
                   for t, d in df.items()}
        return idx

    # ---------------- 检索 ----------------
    def search(self, query: str, top_k: int = 10, *,
               allowed: list[bool] | None = None,
               exclude_ids: set[str] | None = None) -> list[tuple[str, float]]:
        """返回 [(chunk_id, score), ...]，按分数降序。

        Args:
            allowed: 长度为 N 的布尔掩码；None 表示不过滤。
                     在打分前生效 —— 取 top-k 后再筛会让别的公司占满名额。
        """
        q_tokens = tk.tokenize_query(query, bigrams=self.bigrams)
        if not q_tokens or not self.doc_ids:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in set(q_tokens):
            idf = self.idf.get(term)
            if idf is None:              # 词表里没有，直接跳过
                continue
            for doc_i, tf in self.postings[term]:
                if allowed is not None and not allowed[doc_i]:
                    continue
                denom = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_len[doc_i] / (self.avgdl or 1.0))
                scores[doc_i] += idf * tf * (self.k1 + 1) / denom

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        out: list[tuple[str, float]] = []
        for doc_i, s in ranked:
            cid = self.doc_ids[doc_i]
            if exclude_ids and cid in exclude_ids:
                continue
            out.append((cid, s))
            if len(out) >= top_k:
                break
        return out

    # ---------------- 落盘 ----------------
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        p = d / INDEX_NAME
        with open(p, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return p

    @classmethod
    def load(cls, directory: str | Path) -> "BM25Index":
        p = Path(directory) / INDEX_NAME
        if not p.exists():
            raise FileNotFoundError(f"BM25 索引不存在：{p}（先跑 scripts/build_index.py）")
        with open(p, "rb") as f:
            idx = pickle.load(f)
        # k1/b 是**打分参数**，不是索引数据，但整个对象被 pickle 了、它们也跟着
        # 存进了文件。于是改上面的 K1/B 常量之后，老索引会把旧值带回来 ——
        # 表现为"改了参数却毫无效果"，很难查。这里强制以代码里的常量为准。
        # 需要临时换参数（比如调参扫描）在 load() 之后直接改 idx.k1 / idx.b。
        idx.k1 = K1
        idx.b = B
        return idx

    # ---------------- 统计 ----------------
    def stats(self) -> dict:
        lens = self.doc_len or [0]
        return {
            "docs": len(self.doc_ids),
            "terms": len(self.postings),
            "postings": sum(len(v) for v in self.postings.values()),
            "avgdl": round(self.avgdl, 1),
            "min_len": min(lens),
            "max_len": max(lens),
        }
