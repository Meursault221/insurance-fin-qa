"""从 data/chunks 构建检索索引（BM25 + 向量）。

用法：
    python scripts/build_index.py                 # 缺什么建什么
    python scripts/build_index.py --bm25-only     # 只建 BM25（不需要 torch）
    python scripts/build_index.py --force         # 全重建
    python scripts/build_index.py --check         # 只核对现有索引与 chunks 是否一致

两个索引的 doc_ids 必须与 chunks 完全同序 —— 加载时会校验。
chunks 变过（重新抽取/切块）之后索引必然失效，那时要 --force 重建；
不重建不会报错，但会返回"答案对、页码错"的结果，所以 --check 值得常跑。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config as cfg                      # noqa: E402
from src.bm25 import BM25Index                     # noqa: E402
from src.chunker import load_chunks                # noqa: E402
from src.retriever import load_corpus              # noqa: E402

FINGERPRINT = "chunks_fingerprint.json"


def chunk_fingerprint(chunks) -> str:
    """chunks 的内容指纹。索引和 chunks 是否配套，靠它判断。"""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(c.text.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


def load_all_chunks(chunks_dir: Path):
    """读取并过滤语料。

    过滤和检索层用的是同一个函数（retriever.load_corpus）—— 两边分开写的话，
    一旦规则不一致，建出来的索引和检索时加载的 chunks 就对不上，
    而一致性校验会把这种情况拦下来报错，白跑一趟。
    """
    print(f"读取 chunks：{chunks_dir}")
    for p in sorted(chunks_dir.glob("*.jsonl")):
        print(f"  {p.name:36} {len(load_chunks(p)):>6} 条")

    chunks, dropped = load_corpus(chunks_dir)
    if dropped:
        print(f"  已跳过 {len(dropped)} 条非配置公司的 chunk"
              f"（合成样本等）：{dropped[0]}")
    return chunks


def write_fingerprint(directory: Path, fp: str, n: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / FINGERPRINT).write_text(json.dumps(
        {"chunks_fingerprint": fp, "chunks": n},
        ensure_ascii=False, indent=2), encoding="utf-8")


def check_fingerprint(directory: Path, fp: str) -> str:
    """返回 'match' / 'stale' / 'missing'。"""
    p = directory / FINGERPRINT
    if not p.exists():
        return "missing"
    try:
        old = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "missing"
    return "match" if old.get("chunks_fingerprint") == fp else "stale"


def main() -> int:
    ap = argparse.ArgumentParser(description="构建检索索引")
    ap.add_argument("--force", action="store_true", help="索引已存在也重建")
    ap.add_argument("--bm25-only", action="store_true", help="跳过向量索引")
    ap.add_argument("--vector-only", action="store_true", help="跳过 BM25")
    ap.add_argument("--check", action="store_true", help="只核对，不构建")
    ap.add_argument("--batch-size", type=int, default=32, help="向量编码批大小")
    ap.add_argument("--model", default=None, help="覆盖向量模型名")
    args = ap.parse_args()

    cfg.ensure_dirs()
    chunks = load_all_chunks(cfg.CHUNKS_DIR)
    fp = chunk_fingerprint(chunks)

    total_chars = sum(len(c.text) for c in chunks)
    print(f"\n合计 {len(chunks)} 条 chunk，{total_chars:,} 字符，"
          f"指纹 {fp[:16]}…")

    want_bm25 = not args.vector_only
    want_vector = not args.bm25_only
    bm25_fp = check_fingerprint(cfg.BM25_DIR, fp) if want_bm25 else None
    vec_fp = check_fingerprint(cfg.VECTOR_DIR, fp) if want_vector else None

    if args.check:
        print("\n--- 索引核对 ---")
        for name, state, d in (("BM25", bm25_fp, cfg.BM25_DIR),
                               ("向量", vec_fp, cfg.VECTOR_DIR)):
            if state is None:
                continue
            label = {"match": "✅ 与 chunks 配套",
                     "stale": "❌ 已过期（chunks 变过），需 --force 重建",
                     "missing": "⬜ 尚未构建"}[state]
            print(f"  {name:5} {label}")
        return 0 if all(s == "match" for s in (bm25_fp, vec_fp) if s) else 1

    # ---------------------------------------------------------------- BM25
    if want_bm25:
        if bm25_fp == "match" and not args.force:
            print("\nBM25：索引已是最新，跳过（--force 可强制重建）")
        else:
            t0 = time.time()
            print("\nBM25：建索引…")
            idx = BM25Index.build(chunks)
            idx.save(cfg.BM25_DIR)
            write_fingerprint(cfg.BM25_DIR, fp, len(chunks))
            st = idx.stats()
            print(f"  文档 {st['docs']} / 词条 {st['terms']:,} / "
                  f"倒排项 {st['postings']:,} / 平均长度 {st['avgdl']} 词")
            print(f"  最长 {st['max_len']} 词，最短 {st['min_len']} 词，"
                  f"耗时 {time.time() - t0:.1f}s")
            print(f"  → {cfg.BM25_DIR}")

    # -------------------------------------------------------------- 向量
    if want_vector:
        if vec_fp == "match" and not args.force:
            print("\n向量：索引已是最新，跳过（--force 可强制重建）")
        else:
            try:
                from src.vector_index import DEFAULT_MODEL, VectorIndex
            except ImportError as exc:
                print(f"\n向量：跳过 —— {exc}")
                return 1
            t0 = time.time()
            model = args.model or DEFAULT_MODEL
            print(f"\n向量：用 {model} 编码 {len(chunks)} 条 chunk…")
            print("  （首次运行会从 HuggingFace 下载模型，约 95MB）")
            idx = VectorIndex.build(chunks, model_name=model,
                                    batch_size=args.batch_size)
            idx.save(cfg.VECTOR_DIR)
            write_fingerprint(cfg.VECTOR_DIR, fp, len(chunks))
            st = idx.stats()
            print(f"  文档 {st['docs']} / 维度 {st['dim']} / "
                  f"归一化范数 [{st['norm_min']}, {st['norm_max']}]")
            print(f"  耗时 {time.time() - t0:.1f}s")
            print(f"  → {cfg.VECTOR_DIR}")
            if not (abs(st["norm_min"] - 1) < 1e-4 and abs(st["norm_max"] - 1) < 1e-4):
                print("  ⚠ 范数不是 1，向量没归一化，余弦相似度会失真")

    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
