"""Phase 1.5 批量校验 + 汇总报告。

对每家公司做基础验证，产出 `data/samples/phase1_batch_report.json`：
    company / pdf_path / page_count / extracted_path / chunk_path /
    chunk_count / status / warnings

另外针对本作业的硬要求单独核对：

  - **页码是否保留**：chunk.page 必须落在 1..N，且覆盖到每一页
  - **是否有明显数字丢失**：逐页比对 PDF 文字层的数字串与 chunk 文本里的数字串
  - **乱码页**：字体缺 ToUnicode 映射，抽出来是乱码
  - **表格质量分布**：numeric / text / empty

数字比对会**先扣掉页眉页脚里的页码**，否则每页的"15""16"都会算成丢失，
把真正的数据丢失淹没在假阳性里。

**只报告，不修复、不美化。** 发现丢失就把页码和数字原样列出来。

用法：
    python scripts/phase1_batch_report.py
    python scripts/phase1_batch_report.py --only china_life,cpic
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src import chunker, config, pymupdf_compat  # noqa: E402
from src.schema import ExtractedDocument  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples" / "phase1_batch_report.json"

_DIGIT_RUN = re.compile(r"\d[\d,]*(?:\.\d+)?")

# 正常年报的文字层里，数字字符大约占 7%~13%。低于这个线说明
# 财务报表的数字压根没进文字层（被转成了矢量轮廓），见 _scan_vectorized。
MIN_DIGIT_DENSITY = 0.02
# 页面"画了多少东西"的阈值：数字转轮廓会产生大量填充路径
MIN_DRAWINGS = 40


def _runs(text: str) -> set[str]:
    return set(_DIGIT_RUN.findall(text or ""))


_RANK = {"OK": 0, "WARN": 1, "FAIL": 2}


def _escalate(rec: dict, status: str) -> None:
    """状态只升不降 —— 否则后面的 WARN 会把前面的 FAIL 盖掉。"""
    if _RANK[status] > _RANK[rec["status"]]:
        rec["status"] = status


def _scan_vectorized(pdf: Path, ext, fitz, max_detail: int = 30) -> dict:
    """找出"页面上画了东西、文字层里却一个数字都没有"的页。

    中国人保的年报就是这种：合并资产负债表的数字全部转成了矢量轮廓，
    文字层里只剩标签（"货币资金"后面直接是空的）。
    这类丢失用"PDF文字层 vs chunk"的比对**发现不了** —— 两边都没有这些数字。
    只能反过来看：页面上有大量绘图路径，文字层里却没有一个数字。
    """
    doc = fitz.open(pdf)
    hits: list[dict] = []
    for p in ext.pages:
        if not (1 <= p.page <= doc.page_count):
            continue
        if (p.char_count or len(p.text or "")) < 100:      # 封面/纯图页，不算
            continue
        if any(ch.isdigit() for ch in (p.text or "")):
            continue
        n = len(doc[p.page - 1].get_drawings())
        if n >= MIN_DRAWINGS:
            hits.append({"page": p.page, "drawings": n})
    doc.close()
    # 总页数单独给，不要拿截断后的样本长度当结论
    return {"count": len(hits), "pages": [h["page"] for h in hits[:max_detail]]}


def _fmt_char_count(n: int) -> str:
    return f"{n:,}"


def check_company(prof, fitz) -> dict:
    """校验一家公司，返回报告条目。"""
    pdf = ROOT / "data" / "raw" / prof.expected_filename
    doc_id = Path(prof.expected_filename).stem
    ext_path = ROOT / "data" / "extracted" / f"{doc_id}.json"
    chunk_path = ROOT / "data" / "chunks" / f"{doc_id}.jsonl"

    rec: dict = {
        "key": prof.key,
        "company": prof.company,
        "pdf_path": str(pdf.relative_to(ROOT)) if pdf.exists() else "",
        "extracted_path": str(ext_path.relative_to(ROOT)) if ext_path.exists() else "",
        "chunk_path": str(chunk_path.relative_to(ROOT)) if chunk_path.exists() else "",
        "page_count": 0,
        "pdf_page_count": 0,
        "text_chars": 0,
        "chunk_count": 0,
        "chunk_chars": 0,
        "pages_covered": 0,
        "missing_pages": [],
        "missing_digits": {"pages_affected": 0, "count": 0, "samples": []},
        "garbled_pages": [],
        "table_count": 0,
        "table_quality": {},
        "warnings": [],
        "status": "OK",
    }

    if not pdf.exists():
        rec["status"] = "FAIL"
        rec["warnings"].append(f"原始 PDF 不存在：{rec['pdf_path'] or prof.expected_filename}")
        return rec

    # ---- PDF 本身 ----
    try:
        doc = fitz.open(pdf)
    except Exception as exc:
        rec["status"] = "FAIL"
        rec["warnings"].append(f"PDF 打不开：{type(exc).__name__}: {exc}")
        return rec

    n_pdf = doc.page_count
    rec["pdf_page_count"] = n_pdf
    rec["page_count"] = n_pdf
    raw_text = [doc[i].get_text() for i in range(n_pdf)]
    doc.close()

    if prof.page_count and prof.page_count != n_pdf:
        rec["warnings"].append(
            f"配置里记的页数 {prof.page_count} 与实际 {n_pdf} 不符")

    # ---- 抽取结果 ----
    if not ext_path.exists():
        rec["status"] = "FAIL"
        rec["warnings"].append(f"抽取结果不存在：{rec['extracted_path']}")
        return rec

    ext = ExtractedDocument.from_dict(json.loads(ext_path.read_text(encoding="utf-8")))
    rec["page_count"] = len(ext.pages)
    rec["text_chars"] = sum(p.char_count or len(p.text or "") for p in ext.pages)

    digit_chars = sum(sum(ch.isdigit() for ch in (p.text or "")) for p in ext.pages)
    rec["digit_chars"] = digit_chars
    rec["digit_density"] = round(digit_chars / max(1, rec["text_chars"]), 4)
    rec["vectorized_pages"] = []

    if rec["digit_density"] < MIN_DIGIT_DENSITY:
        vec = _scan_vectorized(pdf, ext, fitz)
        rec["vectorized_pages"] = vec["pages"]
        rec["vectorized_page_count"] = vec["count"]
        rec["status"] = "FAIL"
        rec["warnings"].append(
            f"数字密度异常：整份文件文字层里只有 {digit_chars:,} 个数字字符"
            f"（占比 {rec['digit_density']:.3f}，正常年报约 0.07~0.13）。"
            f"{vec['count']} 页有大量矢量绘图但文字层无任何数字"
            f"（例：{vec['pages'][:6]}）—— "
            f"财务报表的数字疑被转为矢量轮廓，文字层取不到，**需要 OCR 才能救回**，"
            f"这家公司的数字目前不可用于问答")

    if len(ext.pages) != n_pdf:
        _escalate(rec, "WARN")
        rec["warnings"].append(f"抽取页数 {len(ext.pages)} 与 PDF 页数 {n_pdf} 不一致")

    # 每页的"页眉页脚数字"，比对时要从 PDF 文字层里扣掉
    footer_runs: dict[int, set[str]] = {}
    for p in ext.pages:
        runs: set[str] = set()
        for blk in p.blocks:
            if blk.kind == "page_footer":
                runs |= _runs(blk.text)
        if runs:
            footer_runs[p.page] = runs

    garbled = [p.page for p in ext.pages if p.warnings]
    rec["garbled_pages"] = garbled
    rec["table_count"] = sum(len(p.tables) for p in ext.pages)

    q: dict[str, int] = {}
    for _pg, tb in ext.iter_tables():
        q[tb.quality] = q.get(tb.quality, 0) + 1
    rec["table_quality"] = q

    # ---- chunks ----
    if not chunk_path.exists():
        rec["status"] = "FAIL"
        rec["warnings"].append(f"chunks 不存在：{rec['chunk_path']}")
        return rec

    chunks = chunker.load_chunks(chunk_path)
    rec["chunk_count"] = len(chunks)
    if not chunks:
        rec["status"] = "FAIL"
        rec["warnings"].append("chunks 文件是空的")
        return rec

    rec["chunk_chars"] = sum(len(c.text or "") for c in chunks)

    # chunk 文本按页归集（跨页 chunk 摊到它覆盖的每一页）
    by_page: dict[int, list[str]] = {}
    bad_page = []
    for c in chunks:
        end = c.page_end or c.page
        if not (1 <= c.page <= n_pdf):
            bad_page.append(c.chunk_id)
        for p in range(c.page, max(c.page, end) + 1):
            by_page.setdefault(p, []).append(c.text or "")
    if bad_page:
        _escalate(rec, "WARN")
        rec["warnings"].append(f"{len(bad_page)} 个 chunk 的页码越界，例：{bad_page[:3]}")

    covered = set(by_page)
    rec["pages_covered"] = len(covered)
    missing_pages = [p for p in range(1, n_pdf + 1) if p not in covered]
    rec["missing_pages"] = missing_pages
    if missing_pages:
        rec["warnings"].append(
            f"{len(missing_pages)} 页没有任何 chunk（可能是纯图片页），"
            f"例：{missing_pages[:8]}")

    # ---- 数字保真：逐页比对 ----
    total_missing = 0
    affected = 0
    samples: list[str] = []
    for p in ext.pages:
        if not (1 <= p.page <= n_pdf):
            continue
        want = _runs(raw_text[p.page - 1]) - footer_runs.get(p.page, set())
        if not want:
            continue
        got: set[str] = set()
        for t in by_page.get(p.page, []):
            got |= _runs(t)
        for tb in p.tables:
            got |= _runs(tb.text)
            for row in tb.cells:
                for cell in row:
                    got |= _runs(cell or "")
        lost = want - got
        if lost:
            affected += 1
            total_missing += len(lost)
            if len(samples) < 8:
                samples.append(f"第{p.page}页 丢失 {sorted(lost)[:6]}")
    rec["missing_digits"] = {
        "pages_affected": affected,
        "count": total_missing,
        "samples": samples,
    }
    if total_missing:
        pct = affected / max(1, len(ext.pages)) * 100
        rec["warnings"].append(
            f"{affected} 页（{pct:.1f}%）存在数字未进入 chunk，共 {total_missing} 个")

    # ---- 来源字段完整性 ----
    bad_src = [c.chunk_id for c in chunks
               if not c.company or not c.report_name or not c.report_year or not c.page]
    if bad_src:
        _escalate(rec, "FAIL")
        rec["warnings"].append(f"{len(bad_src)} 个 chunk 缺少来源字段，例：{bad_src[:3]}")
    no_section = sum(1 for c in chunks if not (c.section or "").strip())
    if no_section:
        rec["warnings"].append(
            f"{no_section} 个 chunk 的 section 为空"
            f"（{no_section / len(chunks) * 100:.1f}%）")

    if rec["warnings"]:
        _escalate(rec, "WARN")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1.5 批量校验")
    ap.add_argument("--only", help="逗号分隔的配置 key")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    _, companies = config.load_companies()
    keys = [k.strip() for k in (args.only or "").split(",") if k.strip()]
    if keys:
        companies = [c for c in companies if c.key in keys]

    fitz = pymupdf_compat.require()
    report = []
    for c in companies:
        if not args.quiet:
            print(f"校验 {c.company} …", flush=True)
        rec = check_company(c, fitz)
        report.append(rec)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 控制台汇总 ----
    print(f"\n{'公司':<6} {'页数':>5} {'chunks':>7} {'文本字符':>10} "
          f"{'数字密度':>8} {'乱码页':>6} {'丢数字页':>8}  状态")
    print("-" * 72)
    for r in report:
        print(f"{r['company']:<6} {r['page_count']:>5} {r['chunk_count']:>7} "
              f"{_fmt_char_count(r['text_chars']):>10} "
              f"{r['digit_density']:>8.3f} "
              f"{len(r['garbled_pages']):>6} "
              f"{r['missing_digits']['pages_affected']:>8}  {r['status']}")

    ok = sum(1 for r in report if r["status"] == "OK")
    warn = sum(1 for r in report if r["status"] == "WARN")
    fail = sum(1 for r in report if r["status"] == "FAIL")
    print("-" * 68)
    print(f"OK {ok} / WARN {warn} / FAIL {fail}，共 {len(report)} 家")
    print(f"报告已写入 {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
