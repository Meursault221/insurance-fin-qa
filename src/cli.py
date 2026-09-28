"""命令行入口。

用法（在项目根目录下执行）:

    python -m src.cli doctor
    python -m src.cli companies
    python -m src.cli extract data/raw/pingan_2025_annual.pdf
    python -m src.cli extract data/raw/pingan_2025_annual.pdf --no-write --preview 3
    python -m src.cli batch
    python -m src.cli preview data/chunks/pingan_2025_annual.jsonl --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import (chunker, config, pdf_extractor, pipeline, pymupdf_compat,
               tables as tables_mod)
from .chunker import ChunkConfig


# ==========================================================================
def cmd_doctor(args) -> int:
    """检查依赖与目录是否就绪。"""
    print("=" * 66)
    print("环境自检")
    print("=" * 66)

    print(f"Python      : {sys.version.split()[0]}")
    print(f"项目根目录  : {config.PROJECT_ROOT}")
    print()

    ok_all = True

    # PyMuPDF
    if pymupdf_compat.is_available():
        print(f"[OK]   PyMuPDF      {pymupdf_compat.version()}")
    else:
        ok_all = False
        print("[FAIL] PyMuPDF      未安装 ->  pip install PyMuPDF")

    # 表格后端
    for name, (ok, msg) in tables_mod.probe_backends().items():
        if ok:
            print(f"[OK]   表格后端 {name}")
        else:
            print(f"[WARN] 表格后端 {name}  {msg}")

    # 可选依赖
    for mod, hint in (("yaml", "pip install PyYAML"),
                      ("pdfplumber", "pip install pdfplumber  (备用表格后端)")):
        try:
            __import__(mod)
            print(f"[OK]   {mod}")
        except ImportError:
            print(f"[WARN] {mod} 未安装 ->  {hint}")

    print()
    print("目录:")
    for d in (config.RAW_DIR, config.EXTRACTED_DIR, config.CHUNKS_DIR,
              config.INDEX_DIR, config.FIXTURES_DIR):
        mark = "OK  " if d.exists() else "缺失"
        print(f"  [{mark}] {d}")
    config.ensure_dirs()

    print()
    n_raw = len(list(config.RAW_DIR.glob('*.pdf'))) if config.RAW_DIR.exists() else 0
    print(f"data/raw 里现有 PDF：{n_raw} 份")
    if n_raw == 0:
        print("  -> 还没有 PDF。可以先跑合成样例（非真实财报）验证流程：")
        print("     python scripts/make_demo_pdf.py")
        print("     python -m src.cli extract tests/fixtures/demo_company_2025_annual.pdf \\")
        print('            --company "示例公司" --key demo --year 2025 \\')
        print('            --report-name "2025年年度报告（合成样本）" --preview 3')

    print("=" * 66)
    return 0 if ok_all else 1


def cmd_companies(args) -> int:
    meta, profiles = config.load_companies()
    print(f"报告年份: {meta.get('report_year')}   类型: {meta.get('report_type')}")
    print("-" * 100)
    print(f"{'key':<16}{'公司':<10}{'市场':<6}{'股票代码':<26}{'报告名':<18}{'核对'}")
    print("-" * 100)
    for p in profiles:
        print(f"{p.key:<16}{p.company:<10}{p.market:<6}"
              f"{','.join(p.stock_codes):<26}{p.report_name:<18}"
              f"{'是' if p.verified else '否 ← 待核对'}")
    print("-" * 100)
    n_unverified = sum(1 for p in profiles if not p.verified)
    if n_unverified:
        print(f"⚠️  有 {n_unverified} 家公司的股票代码尚未核对（verified: false）。")
        print("    下载 PDF 前请到交易所页面确认，并填好 source_url。")
    print(f"\n把 PDF 放到: {config.RAW_DIR}")
    print("文件名建议:")
    for p in profiles:
        print(f"    {p.expected_filename}")
    return 0


def _resolve_meta_args(args):
    """把命令行参数 + companies.yaml 合并成建 meta 需要的字段。"""
    pdf_path = Path(args.pdf)
    profile = None

    key = args.profile or config.guess_key_from_filename(pdf_path.name)
    if key:
        try:
            profile = config.get_profile(key)
        except KeyError:
            if not args.profile:
                profile = None
            else:
                raise

    year = args.year or config.guess_year_from_filename(pdf_path.name, 2025)
    rtype = args.report_type or config.guess_report_type_from_filename(pdf_path.name)

    if profile:
        company = args.company or profile.company
        company_key = args.key or profile.key
        company_full = profile.company_full
        stock_codes = profile.stock_codes
        exchange = profile.exchange
        source_url = profile.source_url
        is_hk = "H" in (profile.market or "")
        report_name = args.report_name or profile.report_name or config.guess_report_name(year, rtype, is_hk)
    else:
        company = args.company
        company_key = args.key or pdf_path.stem.lower().replace(" ", "_")
        company_full = ""
        stock_codes = []
        exchange = ""
        source_url = ""
        report_name = args.report_name or config.guess_report_name(year, rtype)

    if not company:
        raise SystemExit(
            "无法确定公司名。请用 --profile <key> 指定（见 `python -m src.cli companies`），"
            "或用 --company 手动指定。\n"
            "提示：脚本不会瞎猜公司名 —— 来源标错比没有来源更糟。"
        )
    if not company_key:
        raise SystemExit("无法确定 company_key，请用 --key 指定。")
    if not report_name:
        raise SystemExit("无法确定报告名称，请用 --report-name 指定。")

    return dict(
        company=company, company_key=company_key, report_year=year,
        report_name=report_name, company_full=company_full,
        stock_codes=stock_codes, exchange=exchange, report_type=rtype,
        source_url=args.source_url or source_url,
    )


def _print_preview(result: pipeline.SingleResult, n: int) -> None:
    print("\n" + "=" * 66)
    print(f"前 {n} 个 chunk 预览")
    print("=" * 66)
    for c in result.chunks[:n]:
        print(f"\n--- {c.chunk_id}  [{c.block_type}] {c.n_chars} 字 ---")
        print(c.citation())
        body = c.text if len(c.text) <= 500 else c.text[:500] + " …(截断)"
        print("-" * 66)
        print(body)
    print("=" * 66)


def cmd_extract(args) -> int:
    config.ensure_dirs()
    meta_kwargs = _resolve_meta_args(args)

    cfg = ChunkConfig(
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        overlap_chars=args.overlap,
        include_table_text=args.include_table_text,
        keep_tables=not args.no_tables,
    )

    result = pipeline.process_single_pdf(
        args.pdf,
        **meta_kwargs,
        table_backend=args.table_backend,
        table_strategy=args.table_strategy,
        chunk_cfg=cfg,
        write=not args.no_write,
    )

    print(pdf_extractor.quick_report(result.doc))

    cs = result.stats["chunks"]
    print(f"切分结果  : {cs['count']} 个 chunk，"
          f"平均 {cs.get('avg_chars')} 字，"
          f"覆盖 {cs.get('pages_covered')} 页")
    print(f"类型分布  : {cs.get('by_type')}")
    if result.extracted_path:
        print(f"结构化 JSON: {result.extracted_path}")
    if result.chunks_path:
        print(f"chunks     : {result.chunks_path}")
    print(f"耗时       : {result.elapsed_sec}s")

    if args.preview:
        _print_preview(result, args.preview)

    return 0


def cmd_batch(args) -> int:
    config.ensure_dirs()
    cfg = ChunkConfig(
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        overlap_chars=args.overlap,
        include_table_text=args.include_table_text,
        keep_tables=not args.no_tables,
    )
    results = pipeline.process_batch(
        args.pdf_dir,
        only_keys=args.only.split(",") if args.only else None,
        table_backend=args.table_backend,
        table_strategy=args.table_strategy,
        chunk_cfg=cfg,
        write=not args.no_write,
    )
    print(f"\n完成 {len(results)} 份：")
    for r in results:
        cs = r.stats["chunks"]
        print(f"  {r.doc.meta.company:<8} {r.doc.meta.report_name:<14} "
              f"{r.doc.meta.page_count:>4} 页  {cs['count']:>5} chunks  "
              f"表格 {r.doc.stats.get('table_count')}")
    return 0


def cmd_preview(args) -> int:
    chunks = chunker.load_chunks(args.path)
    print(f"共 {len(chunks)} 个 chunk")
    print(json.dumps(chunker.chunk_stats(chunks), ensure_ascii=False, indent=2))
    for c in chunks[: args.limit]:
        print(f"\n--- {c.chunk_id}  [{c.block_type}] ---")
        print(c.citation())
        print(c.text[:400] + (" …" if len(c.text) > 400 else ""))
    return 0


# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="上市保险公司财报问答知识库 —— 第一阶段：PDF 解析与切分",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="检查依赖和目录").set_defaults(func=cmd_doctor)
    sub.add_parser("companies", help="列出 10 家公司配置").set_defaults(func=cmd_companies)

    def add_common_extract_args(sp):
        sp.add_argument("--profile", help="configs/companies.yaml 里的 key，如 pingan")
        sp.add_argument("--company", help="公司中文简称（覆盖配置）")
        sp.add_argument("--key", help="company_key（覆盖自动推断）")
        sp.add_argument("--year", type=int, help="报告年份")
        sp.add_argument("--report-name", help="报告名称，如 '2025年年度报告'")
        sp.add_argument("--report-type", choices=["annual", "semi_annual"])
        sp.add_argument("--source-url", help="真实下载地址（留空也可以，别编）")
        sp.add_argument("--table-backend", default="pymupdf",
                        choices=list(tables_mod.BACKENDS))
        sp.add_argument("--table-strategy", default="auto",
                        choices=list(tables_mod.STRATEGIES),
                        help="PyMuPDF 表格识别策略；中文财报三线表必须用 text（默认）")
        sp.add_argument("--no-tables", action="store_true", help="完全跳过表格")
        sp.add_argument("--include-table-text", action="store_true",
                        help="把表格区域的原始文本行也做成 chunk（默认不，避免与表格重复）")
        sp.add_argument("--max-chars", type=int, default=700)
        sp.add_argument("--min-chars", type=int, default=120)
        sp.add_argument("--overlap", type=int, default=100)
        sp.add_argument("--no-write", action="store_true", help="只跑不落盘")
        sp.add_argument("--preview", type=int, default=0, help="预览前 N 个 chunk")

    sp = sub.add_parser("extract", help="处理单份 PDF")
    sp.add_argument("pdf", help="PDF 路径")
    add_common_extract_args(sp)
    sp.set_defaults(func=cmd_extract)

    sp = sub.add_parser("batch", help="批量处理 data/raw 下所有 PDF")
    sp.add_argument("--pdf-dir", default=str(config.RAW_DIR))
    sp.add_argument("--only", help="只处理这些 key，逗号分隔，如 pingan,cpic")
    add_common_extract_args(sp)
    sp.set_defaults(func=cmd_batch)

    sp = sub.add_parser("preview", help="查看已生成的 chunks")
    sp.add_argument("path", help="chunks JSONL 路径")
    sp.add_argument("--limit", type=int, default=5)
    sp.set_defaults(func=cmd_preview)

    return p


def _force_utf8_stdout() -> None:
    """Windows 上 stdout 被重定向/管道时默认走 GBK(cp936)，中文会乱码。
    统一强制成 UTF-8。现代 Windows Terminal / VS Code 终端都按 UTF-8 解码。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(argv=None) -> int:
    _force_utf8_stdout()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
