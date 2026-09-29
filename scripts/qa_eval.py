"""问答层记录表：跑一批题，把「召回了什么 / 答了什么 / 引用到哪一页」记下来。

作业第四项要求：用至少 10 道题测试，**记录召回了哪些内容、答案是否正确、错在哪里**。
这个脚本负责前一半（机器能自动记的部分），后一半（对不对、错在哪）留给人工填 ——
"答案正确"需要人对着年报核对，脚本不该假装能判。

用真实 API 会产生费用，所以默认从 `configs/eval_questions.yaml` 里取前 10 道，
并且**每道题只调用一次模型**。

用法：
    python scripts/qa_eval.py                          # 前 10 道，写 docs/qa/qa_eval.{json,md}
    python scripts/qa_eval.py -n 29                    # 全部 29 道
    python scripts/qa_eval.py --ids q01,q11,q20        # 指定题号
    python scripts/qa_eval.py --no-llm                 # 只记检索与引用，不调模型（免费）
    python scripts/qa_eval.py --dry-run                # 先看看会花多少次调用，不真跑
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml                                          # noqa: E402

from src import config as cfg                        # noqa: E402
from src.llm import LLMConfig, load_env_file         # noqa: E402
from src.qa import MIN_COSINE, answer_question       # noqa: E402
from src.retriever import MODE_ZH, Retriever         # noqa: E402

QUESTIONS_PATH = cfg.CONFIGS_DIR / "eval_questions.yaml"
OUT_JSON = cfg.PROJECT_ROOT / "docs" / "qa" / "qa_eval.json"
OUT_MD = cfg.PROJECT_ROOT / "docs" / "qa" / "qa_eval.md"


def load_questions(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    qs = data.get("questions") or []
    if not qs:
        raise SystemExit(f"{path} 里没有题目")
    return qs


def pick(questions: list[dict], *, n: int | None, ids: str | None) -> list[dict]:
    if ids:
        want = {s.strip() for s in ids.split(",") if s.strip()}
        got = [q for q in questions if q["id"] in want]
        missing = want - {q["id"] for q in got}
        if missing:
            raise SystemExit(f"找不到题号：{'、'.join(sorted(missing))}")
        return got
    return questions[:n] if n else questions


def run_one(r: Retriever, q: dict, args, cfg_llm: LLMConfig) -> dict:
    """跑一道题，返回一条记录。

    `cfg_llm` 一律显式传下去，**不能传 None**。传 None 的话 `answer_question`
    会回落到 `LLMConfig.from_env()` —— 于是 `--no-llm` 在配了 key 的机器上
    照样会真发请求、真花钱，而且看不出来（它只是想"不接模型"，不是想"用默认配置"）。
    """
    t0 = time.time()
    res = answer_question(q["question"], retriever=r, top_k=args.top_k,
                          mode=args.mode, llm_config=cfg_llm)
    dt = time.time() - t0

    # gold 命中情况：公司 + 页码都要对上，和第二阶段的判分口径一致
    gold_hit = {}
    for t in q.get("targets") or []:
        ranks = [h.rank for h in res.sources
                 if h.company_key == t["company"] and h.page in t["pages"]]
        gold_hit[f"{t['company']}"] = {
            "primary": t.get("primary"),
            "pages": t["pages"],
            "note": t.get("note", ""),
            "hit": bool(ranks),
            "first_rank": min(ranks) if ranks else None,
        }

    return {
        "id": q["id"],
        "question": q["question"],
        "kind": q.get("kind", ""),
        "expect_companies": q.get("expect_companies", []),
        "detected_companies": res.detected_companies,
        "detect_ok": set(res.detected_companies) == set(q.get("expect_companies", [])),
        "top_cosine": round(res.top_cosine, 4),
        "answered_by_llm": res.answered_by_llm,
        "refused": res.refused,
        "refusal_reason": res.refusal_reason,
        "answer": res.answer,
        "notes": res.notes,
        "cited_indexes": res.cited_indexes,
        "suspicious_pages": res.suspicious_pages,
        "gold": gold_hit,
        "gold_hit": all(g["hit"] for g in gold_hit.values()) if gold_hit else None,
        "seconds": round(dt, 1),
        "retrieved": [
            {
                "rank": h.rank, "score": round(h.score, 4), "source": h.source,
                "company_key": h.company_key, "company": h.chunk.company,
                "report_name": h.chunk.report_name, "page": h.chunk.page,
                "section": h.chunk.section, "block_type": h.chunk.block_type,
                "chunk_id": h.chunk.chunk_id,
                "cited": h.rank in res.cited_indexes,
                "text": " ".join(h.chunk.text.split())[:600],
            }
            for h in res.sources
        ],
        # 下面两列留给人工填 —— 脚本判不了"答案对不对"
        "human": {"answer_correct": "", "wrong_at": ""},
    }


def _status_zh(rec: dict) -> str:
    """一句话说清这道题最后怎么了。no_llm 单独一档，别和拒答混。"""
    if rec["answered_by_llm"]:
        return "已生成答案"
    if rec["refusal_reason"] == "no_llm":
        return "未生成答案（没有配置大模型）"
    if rec["refused"]:
        return f"拒答（{rec['refusal_reason']}）"
    return "没有生成答案"


def print_one(rec: dict, show_text: int) -> None:
    print(f"\n{'=' * 72}")
    print(f"{rec['id']}  {rec['question']}")
    print(f"{'-' * 72}")
    print(f"公司识别：{'、'.join(rec['detected_companies']) or '（无）'}"
          f"  {'✅' if rec['detect_ok'] else '❌ 期望 ' + '、'.join(rec['expect_companies'])}")
    print(f"最高相似度：{rec['top_cosine']:.3f}（阈值 {MIN_COSINE:.2f}）")
    # no_llm 不是拒答 —— 是根本没接模型。混在一起会把"没配 key"
    # 说成"系统主动拒绝回答"，那是两回事。
    print(f"结果：{_status_zh(rec)}")
    for note in rec["notes"]:
        print(f"  · {note}")

    print(f"\n召回的 {len(rec['retrieved'])} 条：")
    for r in rec["retrieved"]:
        g = rec["gold"].get(r["company_key"]) or {}
        mark = "★" if (g.get("hit") and r["page"] in (g.get("pages") or [])) else " "
        cite = "*" if r["cited"] else " "
        print(f"  {mark}{cite}[{r['rank']}] {r['company']}｜{r['report_name']}｜"
              f"{r['page']}页｜{r['section'] or '（未识别）'}")
        print(f"        得分 {r['score']:.4f}｜{MODE_ZH.get(r['source'], r['source'])}"
              f"｜{r['block_type']}｜{r['chunk_id']}")
        if show_text:
            print(f"        {r['text'][:show_text]}")

    print(f"\ngold 命中：", end="")
    for key, g in rec["gold"].items():
        flag = f"✅ 首个正确排名 {g['first_rank']}" if g["hit"] else "❌ 没召回"
        print(f"\n  {key}  {flag}   （应含页码 {g['pages']}，主要 {g['primary']}）")
        if g["note"]:
            print(f"     标注依据：{g['note']}")

    print(f"\n答案：\n{rec['answer'] or '（无）'}")
    print(f"\n⏱ {rec['seconds']}s")


def write_md(records: list[dict], args, llm_desc: str) -> None:
    n = len(records)
    gold_ok = sum(1 for r in records if r["gold_hit"])
    detect_ok = sum(1 for r in records if r["detect_ok"])
    # no_llm 不算拒答：那是"没接模型"，不是"系统主动拒绝回答"
    refused = sum(1 for r in records if r["refused"] and r["refusal_reason"] != "no_llm")
    no_llm = sum(1 for r in records if r["refusal_reason"] == "no_llm")
    answered = sum(1 for r in records if r["answered_by_llm"])

    lines = [
        f"# 问答层测试记录（{n} 道题）", "",
        f"- 检索：{MODE_ZH.get(args.mode, args.mode)}，每条问题取 top-{args.top_k} 交给模型",
        f"- 大模型：{llm_desc}",
        f"- 判分口径：**公司 + 页码同时对上**才算召回 gold；"
        f"答案对错由人工核对后填入最后一列",
        "",
        "## 汇总", "",
        "| 指标 | 结果 |",
        "|---|---|",
        f"| 召回 gold 的题目 | {gold_ok}/{n} ({gold_ok / n:.1%}) |",
        f"| 公司识别正确 | {detect_ok}/{n} ({detect_ok / n:.1%}) |",
        f"| 生成了答案 | {answered}/{n} |",
        f"| 拒答（相似度/年份/召回为空） | {refused}/{n} |",
        f"| 未生成（没有配置大模型） | {no_llm}/{n} |",
        "",
        "## 逐题", "",
        "| 题号 | 问题 | 公司识别 | gold 召回 | 状态 | 答案是否正确 | 错在哪里 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        gold = "✅" if r["gold_hit"] else "❌"
        det = "✅" if r["detect_ok"] else "❌"
        if r["refusal_reason"] == "no_llm":
            ref = "未配模型"
        elif r["refused"]:
            ref = f"拒答：{r['refusal_reason']}"
        else:
            ref = "已作答"
        lines.append(f"| {r['id']} | {r['question']} | {det} | {gold} | {ref} "
                     f"| {r['human']['answer_correct']} | {r['human']['wrong_at']} |")

    for r in records:
        lines += ["", f"---", "", f"### {r['id']}　{r['question']}", "",
                  f"- 识别公司：{'、'.join(r['detected_companies']) or '（无）'}"
                  f"　最高相似度 {r['top_cosine']:.3f}",
                  f"- 结果：{_status_zh(r)}",
                  ""]
        if r["notes"]:
            lines += [f"- 提示：{n_}" for n_ in r["notes"]]
            lines.append("")
        lines += ["**召回的片段**", "",
                  "| 排名 | 公司 | 报告 | 页码 | 章节 | 得分 | 被引用 |",
                  "|---|---|---|---|---|---|---|"]
        for x in r["retrieved"]:
            g = r["gold"].get(x["company_key"]) or {}
            star = "★" if (g.get("hit") and x["page"] in (g.get("pages") or [])) else ""
            lines.append(f"| {star}{x['rank']} | {x['company']} | {x['report_name']} "
                         f"| {x['page']} | {x['section'] or '（未识别）'} "
                         f"| {x['score']:.4f} | {'✅' if x['cited'] else ''} |")
        lines += ["", "**答案**", "", "```", r["answer"] or "（无）", "```"]
        if r["suspicious_pages"]:
            lines += ["", f"⚠️ 正文里出现了材料中没有的页码：{r['suspicious_pages']}"]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="问答层记录表")
    ap.add_argument("-n", "--num", type=int, default=10, help="取前 N 道（默认 10）")
    ap.add_argument("--ids", help="指定题号，逗号分隔，如 q01,q11,q20")
    ap.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    ap.add_argument("-k", "--top-k", type=int, default=5)
    ap.add_argument("--mode", default="hybrid",
                    choices=["bm25", "vector", "hybrid"])
    ap.add_argument("--no-llm", action="store_true",
                    help="不调用大模型，只记检索与引用（免费、可反复跑）")
    ap.add_argument("--dry-run", action="store_true", help="只显示要跑哪些题，不执行")
    ap.add_argument("--show-text", type=int, default=0, help="打印片段正文前 N 字")
    ap.add_argument("--json", type=Path, default=OUT_JSON)
    ap.add_argument("--md", type=Path, default=OUT_MD)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    load_env_file()
    llm = LLMConfig.from_env()
    if args.no_llm:
        llm = LLMConfig(provider="none")

    questions = pick(load_questions(args.questions), n=args.num, ids=args.ids)
    llm_desc = (f"{llm.provider} / {llm.model}" if llm.enabled
                else "未配置（只记录检索与引用，不生成答案）")

    print(f"题目：{len(questions)} 道 → {'、'.join(q['id'] for q in questions)}")
    print(f"检索：{MODE_ZH.get(args.mode, args.mode)}  top-{args.top_k}")
    print(f"大模型：{llm_desc}")
    if not args.no_llm and llm.enabled:
        print(f"⚠️ 将发出 {len(questions)} 次大模型调用（可能产生费用）")
    if args.dry_run:
        return 0

    r = Retriever.load()
    records = []
    for q in questions:
        print(f"  跑 {q['id']} …", end="", flush=True)
        rec = run_one(r, q, args, llm)
        print(f" {rec['seconds']}s")
        records.append(rec)

    for rec in records:
        print_one(rec, args.show_text)

    print(f"\n{'=' * 72}")
    gold_ok = sum(1 for x in records if x["gold_hit"])
    detect_ok = sum(1 for x in records if x["detect_ok"])
    print(f"召回 gold：{gold_ok}/{len(records)}    "
          f"公司识别：{detect_ok}/{len(records)}")
    print("（'答案是否正确 / 错在哪里' 两列要在 md 里人工填 —— "
          "脚本判不了，也不该假装能判）")

    if not args.no_write:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "mode": args.mode, "top_k": args.top_k, "num": len(records),
            "llm": llm_desc, "min_cosine": MIN_COSINE, "records": records,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        write_md(records, args, llm_desc)
        print(f"\n完整记录 → {args.json}")
        print(f"记录表   → {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
