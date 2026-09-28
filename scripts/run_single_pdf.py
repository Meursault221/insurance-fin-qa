"""单份 PDF 测试脚本 —— 直接改下面几行配置就能跑。

这是给"我就想快速试一份 PDF"用的最小示例，
演示怎么以编程方式调用 src.pipeline（不用命令行）。

用法:
    # 方式一：改本文件顶部的常量
    python scripts/run_single_pdf.py

    # 方式二：命令行传路径，公司信息自动从文件名/configs 推断
    python scripts/run_single_pdf.py data/raw/pingan_2025_annual.pdf

    # 方式三：完全手动指定来源信息（文件名认不出来时）
    python scripts/run_single_pdf.py my.pdf --company 中国平安 --key pingan \
        --year 2025 --report-name "2025年年度报告"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows 管道下 stdout 默认 GBK，中文会乱码
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src import chunker, config, pdf_extractor, pipeline  # noqa: E402
from src.chunker import ChunkConfig  # noqa: E402

# ==========================================================================
# 【方式一】直接改这里，然后 python scripts/run_single_pdf.py
# ==========================================================================
PDF_PATH = ""            # 例如: "data/raw/pingan_2025_annual.pdf"
COMPANY = ""             # 例如: "中国平安"
COMPANY_KEY = ""         # 例如: "pingan"
REPORT_YEAR = 2025
REPORT_NAME = ""         # 例如: "2025年年度报告"
TABLE_BACKEND = "pymupdf"   # pymupdf | pdfplumber | none
PREVIEW_N = 5            # 打印前几个 chunk
# ==========================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description="单份 PDF 解析测试")
    ap.add_argument("pdf", nargs="?", default=PDF_PATH, help="PDF 路径")
    ap.add_argument("--company", default=COMPANY)
    ap.add_argument("--key", default=COMPANY_KEY)
    ap.add_argument("--year", type=int, default=REPORT_YEAR)
    ap.add_argument("--report-name", default=REPORT_NAME)
    ap.add_argument("--table-backend", default=TABLE_BACKEND,
                    choices=["pymupdf", "pdfplumber", "none"])
    ap.add_argument("--preview", type=int, default=PREVIEW_N)
    args = ap.parse_args()

    if not args.pdf:
        print("请指定 PDF 路径：python scripts/run_single_pdf.py <路径>")
        print("或者先跑合成样本练手：python scripts/make_demo_pdf.py")
        return 2

    pdf = Path(args.pdf)
    if not pdf.is_absolute():
        pdf = config.PROJECT_ROOT / pdf
    if not pdf.exists():
        print(f"找不到文件：{pdf}")
        return 2

    config.ensure_dirs()

    # ---- 来源信息：优先用显式参数，其次查 configs/companies.yaml ----
    profile = None
    key = args.key or config.guess_key_from_filename(pdf.name)
    if key:
        try:
            profile = config.get_profile(key)
        except KeyError:
            profile = None

    company = args.company or (profile.company if profile else "")
    if not company:
        print("无法确定公司名。请加 --company，或把文件名改成配置里约定的形式：")
        print("  python scripts/run_single_pdf.py <pdf> --company 中国平安 --key pingan")
        print("（脚本不会瞎猜公司名 —— 来源标错比没有来源更糟）")
        return 2

    year = args.year
    report_name = args.report_name or (profile.report_name if profile else "")
    if not report_name:
        report_name = config.guess_report_name(year, "annual")

    print(f"PDF        : {pdf}")
    print(f"公司       : {company}")
    print(f"报告       : {report_name}")
    print(f"年份       : {year}")
    print(f"表格后端   : {args.table_backend}")
    print()

    result = pipeline.process_single_pdf(
        pdf,
        company=company,
        company_key=key or pdf.stem.lower(),
        report_year=year,
        report_name=report_name,
        company_full=profile.company_full if profile else "",
        stock_codes=profile.stock_codes if profile else [],
        exchange=profile.exchange if profile else "",
        source_url=profile.source_url if profile else "",
        table_backend=args.table_backend,
        chunk_cfg=ChunkConfig(),
    )

    print(pdf_extractor.quick_report(result.doc))
    cs = result.stats["chunks"]
    print(f"chunks     : {cs['count']} 个（平均 {cs.get('avg_chars')} 字，"
          f"覆盖 {cs.get('pages_covered')} 页）")
    print(f"类型分布   : {cs.get('by_type')}")
    print(f"结构化 JSON: {result.extracted_path}")
    print(f"chunks 文件: {result.chunks_path}")
    print(f"耗时       : {result.elapsed_sec}s")

    if args.preview:
        print("\n" + "=" * 66)
        print(f"前 {args.preview} 个 chunk（含来源信息）")
        print("=" * 66)
        for c in result.chunks[: args.preview]:
            print(f"\n--- {c.chunk_id}  [{c.block_type}]  {c.n_chars} 字 ---")
            print(c.citation())
            body = c.text if len(c.text) <= 400 else c.text[:400] + " …"
            print("-" * 66)
            print(body)
        print("=" * 66)

    print("\n人工抽查建议：")
    print(f"  1. 打开 {result.extracted_path.name}，搜一个你记得的数字，")
    print("     确认 page 字段和 PDF 阅读器里显示的页一致")
    print("  2. 看 tables[].cells 的行列有没有错位")
    print("  3. 看 blocks[].section 的章节路径对不对")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
