"""向量语义检索。

用 sentence-transformers 加载中文检索模型，把 chunk 编码成稠密向量，
检索时算余弦相似度（向量已 L2 归一化，所以点积即余弦）。

模型选 BAAI/bge-small-zh-v1.5：
  - 中文检索效果在同类小模型里靠前，512 维，磁盘约 95MB
  - 体量小，CPU 上编码 7507 个 chunk 在一两分钟内能跑完

两个容易踩的坑，都在代码里处理了：

1. **BGE 的 query 需要加指令前缀**。bge-*-zh-v1.5 系列在检索任务上要求
   给 query 加"为这个句子生成表示以用于检索相关文章："，passage 不加。
   漏掉这一步召回率会明显下降，而且不报错 —— 属于静默变差。
2. **建索引和检索必须用同一个模型**。meta 里记下模型名，加载时校验，
   避免换了模型之后新旧向量混在一个索引里（余弦值会失去意义）。

向量只用于召回，不产生任何文本 —— chunk 原文始终来自 data/chunks，
所以引用里的公司/页码不会因为向量检索而失真。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .tokenize import to_simplified

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

# bge-zh-v1.5 官方要求的检索指令前缀，只加在 query 上
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

META_NAME = "vector_meta.json"
EMB_NAME = "embeddings.npy"


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:      # pragma: no cover
        raise ImportError(
            "缺少 numpy。安装：python -m pip install numpy") from exc
    return np


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    """加载并缓存模型。

    加 lru_cache 是必须的，不只是优化：检索时每条 query 都要编码一次，
    评测跑几百条查询。每次都 new 一个 SentenceTransformer 会重复读盘、
    重复初始化权重，慢到评测根本跑不完。
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:      # pragma: no cover
        raise ImportError(
            "缺少 sentence-transformers。安装：\n"
            "  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch\n"
            "  python -m pip install sentence-transformers") from exc
    return SentenceTransformer(model_name)


@dataclass
class VectorIndex:
    doc_ids: list[str] = field(default_factory=list)
    embeddings: object = None            # numpy.ndarray, (N, dim), L2 归一化
    model_name: str = DEFAULT_MODEL
    dim: int = 0

    # ---------------- 建索引 ----------------
    @classmethod
    def build(cls, chunks, *, model_name: str = DEFAULT_MODEL,
              batch_size: int = 32, progress: bool = True,
              simplify: bool = True,
              text_of=lambda c: c.embed_text) -> "VectorIndex":
        """编码所有 chunk。

        Args:
            text_of: 取哪段文本编码。默认 embed_text —— 它把
                     "【公司·报告名】第N页 章节路径" 拼在正文前面，
                     让向量带上出处信息，问"中国平安的总资产"时更容易命中平安的段落。
            simplify: 编码前转简体。5 家港股年报正文是繁体（占语料 52%），
                     查询多半是简体 —— 两侧表示不一致会白白拉低相似度。
                     和 BM25 用同一套归一，两路召回才对得齐。
        """
        np = _require_numpy()
        model = _load_model(model_name)

        idx = cls(doc_ids=[c.chunk_id for c in chunks], model_name=model_name)
        texts = [text_of(c) for c in chunks]
        if simplify:
            texts = [to_simplified(t) for t in texts]
        emb = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=progress,
            convert_to_numpy=True,
        )
        idx.embeddings = np.asarray(emb, dtype="float32")
        idx.dim = int(idx.embeddings.shape[1])
        return idx

    # ---------------- 检索 ----------------
    def embed_query(self, query: str):
        """把 query 编码成向量。指令前缀在这里加，别漏。"""
        np = _require_numpy()
        model = _load_model(self.model_name)
        v = model.encode([QUERY_INSTRUCTION + to_simplified(query)],
                         normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(v, dtype="float32")[0]

    def search(self, query: str, top_k: int = 10, *,
               allowed: list[bool] | None = None,
               exclude_ids: set[str] | None = None) -> list[tuple[str, float]]:
        """返回 [(chunk_id, cosine), ...]，降序。

        allowed 过滤在算相似度之后、取 top-k 之前做 —— 与 BM25 保持一致的语义。
        """
        np = _require_numpy()
        if self.embeddings is None or not self.doc_ids:
            return []

        qv = None
        if isinstance(query, str):
            qv = self.embed_query(query)
        else:                            # 允许直接传已编码好的向量（评测时复用）
            qv = np.asarray(query, dtype="float32")

        sims = self.embeddings @ qv            # 归一化后点积即余弦
        if allowed is not None:
            mask = np.asarray(allowed, dtype=bool)
            sims = np.where(mask, sims, -np.inf)

        k = min(top_k + (len(exclude_ids) if exclude_ids else 0), len(sims))
        if k <= 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]

        out: list[tuple[str, float]] = []
        for i in top:
            if not np.isfinite(sims[i]):       # 被掩码排除的
                continue
            cid = self.doc_ids[int(i)]
            if exclude_ids and cid in exclude_ids:
                continue
            out.append((cid, float(sims[i])))
            if len(out) >= top_k:
                break
        return out

    # ---------------- 落盘 ----------------
    def save(self, directory: str | Path) -> Path:
        np = _require_numpy()
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / EMB_NAME, self.embeddings)
        (d / META_NAME).write_text(json.dumps({
            "doc_ids": self.doc_ids,
            "model_name": self.model_name,
            "dim": self.dim,
            "count": len(self.doc_ids),
        }, ensure_ascii=False), encoding="utf-8")
        return d / EMB_NAME

    @classmethod
    def load(cls, directory: str | Path,
             model_name: str | None = None) -> "VectorIndex":
        np = _require_numpy()
        d = Path(directory)
        meta_p, emb_p = d / META_NAME, d / EMB_NAME
        if not (meta_p.exists() and emb_p.exists()):
            raise FileNotFoundError(
                f"向量索引不存在：{d}（先跑 scripts/build_index.py）")

        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if model_name and model_name != meta["model_name"]:
            raise ValueError(
                f"索引是用 {meta['model_name']} 建的，不能用 {model_name} 检索 —— "
                f"不同模型的向量不在同一个空间里，余弦值没有意义")
        return cls(doc_ids=meta["doc_ids"], embeddings=np.load(emb_p),
                   model_name=meta["model_name"], dim=int(meta["dim"]))

    # ---------------- 统计 ----------------
    def stats(self) -> dict:
        np = _require_numpy()
        if self.embeddings is None:
            return {"docs": 0}
        norms = np.linalg.norm(self.embeddings, axis=1)
        return {
            "docs": len(self.doc_ids),
            "dim": self.dim,
            "model": self.model_name,
            "dtype": str(self.embeddings.dtype),
            # 归一化后范数应全为 1；不是 1 说明存盘/加载环节出了问题
            "norm_min": round(float(norms.min()), 6),
            "norm_max": round(float(norms.max()), 6),
        }
