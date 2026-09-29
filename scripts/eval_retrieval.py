"""检索层评测：逐题对比 BM25 / 向量 / 混合三路召回。

对每道题、每一路，记录：
  - 召回了哪些 chunk（公司、页码、章节）
  - 正确来源是否被召回（gold 页码 + gold 公司是否出现在 top-k 里）
  - 公司是否正确（自动识别出的公司 vs 题目期望）
  - 首个正确结果的排名（算 MRR）

判定"命中"必须**公司和页码同时对上**。只看页码不够 —— 中国人保和中国财险
的页码会撞车（两份年报都几百页），只比页码会把两家的答案算成同一个。

用法：
    python scripts/eval_retrieval.py                    # 默认 top-10
    python scripts/eval_retrieval.py -k 5 --verbose      # 看每题召回明细
    python scripts/eval_retrieval.py --json docs/qa/retrieval_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml                                          # noqa: E402

from src import config as cfg                        # noqa: E402
from src.retriever import MODE_BM25, MODE_HYBRID, MODE_VECTOR, Retriever  # noqa: E402

MODES_ORDER = (MODE_BM25, MODE_VECTOR, MODE_HYBRID)
MODE_ZH = {MODE_BM25: "BM25", MODE_VECTOR: "向量", MODE_HYBRID: "混合"}

QUESTIONS_PATH = cfg.CONFIGS_DIR / "eval_questions.yaml"


def _w(s: str) -> int:
    """显示宽度。中文占 2 格，否则列头和数据行对不齐。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def _pad(s: str, width: int, right: bool = True) -> str:
    """按显示宽度补齐。right=True 右对齐（数字），False 左对齐（中文标签）。"""
    pad = " " * max(width - _w(str(s)), 0)
    return (pad + str(s)) if right else (str(s) + pad)

# 融合配置候选。--sweep 会把它们在同一批题目上跑一遍做对比。
#
# 这一串是**实测过**的候选，不是理论清单。结论（29 题、top-10）：
#   rrf pool=50 1:1 与 union pool=50 并列最好（26/29、目标召回 90.3%）；
#   加权（2:1 / 3:1）和缩小 pool 都更差；weighted 的 MRR 略高但命中更少。
#   rrf 最简单、最好解释，所以默认取它。
#   还试过"每路留固定席位"的 quota 融合，同样没赢，而且会把 rrf 本来答对的
#   q09/q10/q15 弄丢 —— 那版实现已经删掉，不留在代码里。
# 详见 docs/phase2_retrieval.md。
SWEEP = [
    {"label": "rrf      pool=50  1:1（默认）", "pool": 50, "fusion": "rrf"},
    {"label": "rrf      pool=10  1:1", "pool": 10, "fusion": "rrf"},
    {"label": "rrf      pool=50  2:1（偏 BM25）", "pool": 50, "fusion": "rrf",
     "weights": {"bm25": 2.0, "vector": 1.0}},
    {"label": "rrf      pool=50  3:1（偏 BM25）", "pool": 50, "fusion": "rrf",
     "weights": {"bm25": 3.0, "vector": 1.0}},
    {"label": "rrf      pool=50  1:2（偏向量）", "pool": 50, "fusion": "rrf",
     "weights": {"bm25": 1.0, "vector": 2.0}},
    {"label": "union    pool=50", "pool": 50, "fusion": "union"},
    {"label": "union    pool=50  2:1", "pool": 50, "fusion": "union",
     "weights": {"bm25": 2.0, "vector": 1.0}},
    {"label": "weighted pool=50", "pool": 50, "fusion": "weighted"},
]


def load_questions(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    qs = data.get("questions") or []
    if not qs:
        raise SystemExit(f"{path} 里没有题目")
    for q in qs:
        targets = q.get("targets") or []
        if not targets:
            raise SystemExit(f"题目 {q.get('id')} 没有 targets，无法判定对错")
        for t in targets:
            if not t.get("pages"):
                raise SystemExit(f"题目 {q.get('id')} 的 target 缺 pages")
    return qs


def score_question(retrieved: list, question: dict) -> dict:
    """判分。返回每题每路的明细。"""
    targets = question["targets"]
    hit_ids: set[int] = set()
    first_rank: int | None = None

    for r in retrieved:
        for i, t in enumerate(targets):
            if i in hit_ids:
                continue
            # 公司 + 页码都要对上
            if r.company_key == t["company"] and r.page in t["pages"]:
                hit_ids.add(i)
                if first_rank is None or r.rank < first_rank:
                    first_rank = r.rank

    return {
        "targets_total": len(targets),
        "targets_hit": len(hit_ids),
        "success": len(hit_ids) == len(targets),   # 全部 target 都召回才算这道题成功
        "first_hit_rank": first_rank,
        # 公司判对：命中的结果里，公司确实是 gold 的公司（由命中条件保证）
        "hit_companies": sorted({targets[i]["company"] for i in hit_ids}),
        "recalled": [
            {"rank": r.rank, "score": round(r.score, 4),
             "chunk_id": r.chunk_id, "company": r.company_key,
             "page": r.page, "section": r.chunk.section,
             "block_type": r.chunk.block_type}
            for r in retrieved
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="检索层评测")
    ap.add_argument("-k", "--top-k", type=int, default=10)
    ap.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    ap.add_argument("--verbose", action="store_true", help="打印每题召回明细")
    ap.add_argument("--json", type=Path, default=None, help="把完整结果写成 JSON")
    ap.add_argument("--md", type=Path, default=None, help="把汇总写成 Markdown 表")
    ap.add_argument("--pool", type=int, default=50, help="融合前每路取多少条")
    ap.add_argument("--fusion", default="rrf",
                    choices=["rrf", "weighted", "union"],
                    help="混合检索的融合方式（默认 rrf，由 --sweep 选出）")
    ap.add_argument("--sweep", action="store_true",
                    help="把多种混合融合配置跑一遍做对比")
    args = ap.parse_args()

    t0 = time.time()
    print("加载 chunks 与索引…")
    r = Retriever.load()
    st = r.stats()
    print(f"  {st['chunks']} 条 chunk / {st['companies']} 家公司")
    for k, n in st["by_company"].items():
        print(f"    {k:16} {n:>5} 条，覆盖 {st['pages_covered'][k]:>4} 页")
    if st["bm25"]:
        print(f"  BM25: {st['bm25']['terms']:,} 词条 / "
              f"{st['bm25']['postings']:,} 倒排项 / 平均 {st['bm25']['avgdl']} 词")
    if st["vector"]:
        print(f"  向量: {st['vector']['dim']} 维 / {st['vector']['model']}")

    questions = load_questions(args.questions)
    print(f"\n题目 {len(questions)} 道，top-k = {args.top_k}\n")

    per_q: list[dict] = []
    agg = {m: {"success": 0, "targets_hit": 0, "targets_total": 0,
               "mrr_sum": 0.0, "mrr_n": 0, "top1": 0} for m in MODES_ORDER}
    detect_ok = 0

    for q in questions:
        qid, text = q["id"], q["question"]
        expect = list(q.get("expect_companies") or [])

        res = r.search_per_source(text, args.top_k, pool=args.pool,
                                  fusion=args.fusion)

        # 公司识别：题目期望的公司集合 vs 自动识别出来的
        detected = r.detect_companies(text)
        if set(expect) == set(detected):
            detect_ok += 1

        row = {"id": qid, "question": text, "kind": q.get("kind", ""),
               "expect_companies": expect, "detected_companies": detected,
               "detect_ok": set(expect) == set(detected), "modes": {}}

        for mode in MODES_ORDER:
            s = score_question(res[mode], q)
            row["modes"][mode] = s
            a = agg[mode]
            a["success"] += int(s["success"])
            a["targets_hit"] += s["targets_hit"]
            a["targets_total"] += s["targets_total"]
            if s["first_hit_rank"]:
                a["mrr_sum"] += 1.0 / s["first_hit_rank"]
                a["mrr_n"] += 1
            # 排第一的那条就是正确来源 —— 这是最苛刻的一档：
            # 它决定了答案能不能直接引用第一条，还是得人工往下翻
            if s["first_hit_rank"] == 1:
                a["top1"] += 1

        per_q.append(row)

    # ------------------------------------------------------------- 逐题输出
    n = len(questions)
    print("=" * 100)
    print(_pad("题号", 7, right=False) + _pad("题型", 10, right=False)
          + "".join(_pad(MODE_ZH[m], 13, right=False) for m in MODES_ORDER)
          + "  公司识别")
    print("-" * 100)
    kind_zh = {"exact": "精确", "semantic": "语义", "cross": "跨公司"}
    for row in per_q:
        marks = []
        for m in MODES_ORDER:
            s = row["modes"][m]
            if s["success"]:
                marks.append("✅ 全中")
            elif s["targets_hit"]:
                marks.append(f"◐ {s['targets_hit']}/{s['targets_total']}")
            else:
                marks.append("❌ 未中")
        d = "✅" if row["detect_ok"] else f"❌ 期望{row['expect_companies']}"
        print(_pad(row["id"], 7, right=False)
              + _pad(kind_zh.get(row["kind"], row["kind"]), 10, right=False)
              + "".join(_pad(mk, 13, right=False) for mk in marks) + "  " + d)

    if args.verbose:
        print("\n" + "=" * 100)
        print("逐题召回明细")
        for row in per_q:
            print(f"\n【{row['id']}】{row['question']}")
            for m in MODES_ORDER:
                s = row["modes"][m]
                flag = "命中" if s["success"] else f"{s['targets_hit']}/{s['targets_total']}"
                print(f"  {MODE_ZH[m]}（{flag}，首个正确排名 {s['first_hit_rank']}）：")
                for h in s["recalled"]:
                    print(f"     {h['rank']:>2}. {h['score']:>8.4f}  "
                          f"{h['company']:<15} p{h['page']:<4} "
                          f"[{h['block_type']}] {(h['section'] or '')[:22]}")

    # ------------------------------------------------------------- 汇总指标
    print("\n" + "=" * 100)
    print("汇总")
    print("-" * 100)
    print(_pad("", 10, right=False) + _pad("题目命中率", 14) + _pad("目标召回率", 14)
          + _pad("MRR", 9) + _pad("Top-1命中", 12))
    summary = {}
    for m in MODES_ORDER:
        a = agg[m]
        hit_rate = a["success"] / n
        tgt_rate = a["targets_hit"] / max(a["targets_total"], 1)
        mrr = a["mrr_sum"] / n            # 未命中的题按 0 计
        summary[m] = {"hit_rate": round(hit_rate, 4),
                      "success": a["success"],
                      "target_recall": round(tgt_rate, 4),
                      "mrr": round(mrr, 4),
                      "top1": a["top1"],
                      "top1_rate": round(a["top1"] / n, 4)}
        print(_pad(MODE_ZH[m], 10, right=False)
              + _pad(f"{a['success']}/{n}", 14)
              + _pad(f"{tgt_rate:.1%}", 14) + _pad(f"{mrr:.3f}", 9)
              + _pad(f"{a['top1']}/{n}", 12))
    print(f"\n公司识别正确：{detect_ok}/{n} = {detect_ok / n:.1%}")
    print(f"耗时 {time.time() - t0:.1f}s")

    # 按题型拆分，看 BM25 和向量各自的主场
    print("\n按题型拆分（题目命中率）")
    print("-" * 100)
    by_kind: dict[str, dict[str, int]] = {}
    for row in per_q:
        k = row["kind"]
        by_kind.setdefault(k, {"n": 0, **{m: 0 for m in MODES_ORDER}})
        by_kind[k]["n"] += 1
        for m in MODES_ORDER:
            by_kind[k][m] += int(row["modes"][m]["success"])
    print(_pad("题型", 10, right=False) + _pad("题数", 6)
          + "".join(_pad(MODE_ZH[m], 16) for m in MODES_ORDER))
    for k, v in by_kind.items():
        cells = "".join(
            _pad(f"{v[m]}/{v['n']} ({v[m] / v['n']:.0%})", 16)
            for m in MODES_ORDER)
        print(_pad(kind_zh.get(k, k), 10, right=False) + _pad(str(v["n"]), 6)
              + cells)

    # --------------------------------------------------- 融合配置对比（可选）
    # 混合检索的配置（用哪种融合、每路取多少、怎么加权）不该拍脑袋定。
    # 这里把候选配置在同一批题目上跑一遍，用"目标召回率"挑。
    if args.sweep:
        print("\n" + "=" * 100)
        print("混合检索配置对比（同一批题目，只改融合设置）")
        print("-" * 100)
        print(_pad("配置", 34, right=False) + _pad("题目命中率", 12)
              + _pad("目标召回率", 14) + _pad("MRR", 9) + "  失败题")
        sweep_rows = []
        for cfg in SWEEP:
            hit = tgt_hit = tgt_tot = 0
            mrr = 0.0
            fails: list[str] = []
            for q in questions:
                res = r.search(q["question"], args.top_k,
                               pool=cfg["pool"], fusion=cfg["fusion"],
                               weights=cfg.get("weights"))
                s = score_question(res, q)
                hit += int(s["success"])
                tgt_hit += s["targets_hit"]
                tgt_tot += s["targets_total"]
                if s["first_hit_rank"]:
                    mrr += 1.0 / s["first_hit_rank"]
                if not s["success"]:
                    fails.append(q["id"])
            row = {"label": cfg["label"], "pool": cfg["pool"],
                   "fusion": cfg["fusion"], "weights": cfg.get("weights"),
                   "success": hit, "hit_rate": round(hit / n, 4),
                   "target_recall": round(tgt_hit / max(tgt_tot, 1), 4),
                   "mrr": round(mrr / n, 4), "failed": fails}
            sweep_rows.append(row)
            # 失败题一并打出来 —— 只看汇总数字会以为几种配置"差不多"，
            # 实际上它们丢的往往不是同一道题，换配置是把一类失败换成另一类。
            print(_pad(cfg["label"], 34, right=False) + _pad(f"{hit}/{n}", 12)
                  + _pad(f"{row['target_recall']:.1%}", 14)
                  + _pad(f"{row['mrr']:.3f}", 9) + "  " + " ".join(fails))
        if args.json:
            data = json.loads(args.json.read_text(encoding="utf-8"))
            data["sweep"] = sweep_rows
            args.json.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
            print(f"\n对比结果已并入 {args.json}")

    # ------------------------------------------------------------- 落盘
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "top_k": args.top_k, "pool": args.pool,
            "questions": n,
            "index": {"chunks": st["chunks"], "companies": st["companies"]},
            "summary": summary,
            "detect_accuracy": round(detect_ok / n, 4),
            "by_kind": by_kind,
            "details": per_q,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n完整结果 → {args.json}")

    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# 检索层评测结果（top-{args.top_k}）", "",
                 f"- 索引：{st['chunks']} 条 chunk，{st['companies']} 家公司",
                 f"- 题目：{n} 道，覆盖 {len({c for q in questions for c in q['expect_companies']})} 家公司",
                 "",
                 "## 汇总", "",
                 "| 检索方式 | 题目命中率 | 目标召回率 | MRR | Top-1 命中 |",
                 "|---|---|---|---|---|"]
        for m in MODES_ORDER:
            s = summary[m]
            lines.append(f"| {MODE_ZH[m]} | {s['success']}/{n} ({s['hit_rate']:.1%}) "
                         f"| {s['target_recall']:.1%} | {s['mrr']:.3f} "
                         f"| {s['top1']}/{n} ({s['top1_rate']:.1%}) |")
        lines += ["", f"公司识别正确率：{detect_ok}/{n} = {detect_ok / n:.1%}", "",
                  "## 逐题结果", "",
                  "| 题号 | 题型 | 问题 | BM25 | 向量 | 混合 | 公司识别 |",
                  "|---|---|---|---|---|---|---|"]
        for row in per_q:
            cells = []
            for m in MODES_ORDER:
                s = row["modes"][m]
                cells.append("✅" if s["success"]
                             else (f"◐{s['targets_hit']}/{s['targets_total']}"
                                   if s["targets_hit"] else "❌"))
            lines.append(f"| {row['id']} | {kind_zh.get(row['kind'], row['kind'])} "
                         f"| {row['question']} | " + " | ".join(cells)
                         + f" | {'✅' if row['detect_ok'] else '❌'} |")
        args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"汇总表 → {args.md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
