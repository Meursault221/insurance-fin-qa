"""编排层：把 PDF -> 结构化 JSON -> chunks 串起来。

对外只暴露两个函数：
    process_single_pdf(...)   单份 PDF（第一阶段的主入口）
    process_batch(...)        批量（第 10 家公司齐了之后用，现在先留好接口）
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from . import chunker, config, pdf_extractor
from .chunker import ChunkConfig
from .schema import Chunk, ExtractedDocument


@dataclass
class SingleResult:
    doc: ExtractedDocument
    chunks: list[Chunk]
    extracted_path: Optional[Path] = None
    chunks_path: Optional[Path] = None
    elapsed_sec: float = 0.0
    stats: dict = field(default_factory=dict)


def process_single_pdf(
    pdf_path: str | Path,
    *,
    company: str,
    company_key: str,
    report_year: int,
    report_name: str,
    company_full: str = "",
    stock_codes: Optional[list[str]] = None,
    exchange: str = "",
    report_type: str = "annual",
    source_url: str = "",
    table_backend: str = "pymupdf",
    table_strategy: str = "auto",
    chunk_cfg: Optional[ChunkConfig] = None,
    out_extracted: Optional[Path] = None,
    out_chunks: Optional[Path] = None,
    write: bool = True,
) -> SingleResult:
    """处理一份 PDF：抽取 -> 落 JSON -> 切分 -> 落 JSONL。"""
    t0 = _dt.datetime.now()

    meta = pdf_extractor.build_meta(
        pdf_path,
        company=company,
        company_key=company_key,
        report_year=report_year,
        report_name=report_name,
        company_full=company_full,
        stock_codes=stock_codes,
        exchange=exchange,
        report_type=report_type,
        source_url=source_url,
    )

    doc = pdf_extractor.extract_pdf(
        pdf_path, meta,
        table_backend=table_backend,
        table_strategy=table_strategy,
    )

    ext_path = None
    if write:
        ext_path = pdf_extractor.save_extracted(
            doc, out_extracted or config.EXTRACTED_DIR)

    chunks = chunker.chunk_document(doc, chunk_cfg or ChunkConfig())

    chk_path = None
    if write:
        target_dir = Path(out_chunks) if out_chunks else config.CHUNKS_DIR
        chk_path = target_dir / f"{meta.doc_id}.jsonl"
        chunker.save_chunks(chunks, chk_path)

    elapsed = (_dt.datetime.now() - t0).total_seconds()

    return SingleResult(
        doc=doc,
        chunks=chunks,
        extracted_path=ext_path,
        chunks_path=chk_path,
        elapsed_sec=round(elapsed, 2),
        stats={
            "extract": doc.stats,
            "chunks": chunker.chunk_stats(chunks),
            "elapsed_sec": round(elapsed, 2),
        },
    )


def process_batch(
    pdf_dir: str | Path,
    *,
    only_keys: Optional[Iterable[str]] = None,
    **kwargs,
) -> list[SingleResult]:
    """批量处理 data/raw 下的所有 PDF。

    文件名必须能被 config.guess_key_from_filename 认出来，
    否则会被跳过并列在 skipped 里（不会瞎猜公司名）。
    """
    pdf_dir = Path(pdf_dir)
    results: list[SingleResult] = []
    skipped: list[tuple[str, str]] = []

    only = set(only_keys) if only_keys else None

    # Windows 上 glob 大小写不敏感，"*.pdf" 和 "*.PDF" 会命中同一批文件，
    # 直接拼起来会让每份 PDF 被处理两次。这里按路径去重。
    seen: set[Path] = set()
    pdfs: list[Path] = []
    for p in sorted(pdf_dir.glob("*")):
        if p.suffix.lower() == ".pdf" and p not in seen:
            seen.add(p)
            pdfs.append(p)
    pdfs.sort()

    for pdf in pdfs:
        key = config.guess_key_from_filename(pdf.name)
        if key is None:
            skipped.append((pdf.name, "文件名无法匹配 configs/companies.yaml 里的公司"))
            continue
        if only and key not in only:
            continue

        profile = config.get_profile(key)
        year = config.guess_year_from_filename(pdf.name, 2025)
        rtype = config.guess_report_type_from_filename(pdf.name)

        try:
            res = process_single_pdf(
                pdf,
                company=profile.company,
                company_key=profile.key,
                report_year=year,
                report_name=profile.report_name or config.guess_report_name(
                    year, rtype, is_hk=("H" in profile.market)),
                company_full=profile.company_full,
                stock_codes=profile.stock_codes,
                exchange=profile.exchange,
                report_type=rtype,
                source_url=profile.source_url,
                **kwargs,
            )
            results.append(res)
        except Exception as exc:  # 单份失败不影响整批
            skipped.append((pdf.name, f"处理失败: {exc}"))

    if skipped:
        print("\n[跳过]")
        for name, why in skipped:
            print(f"  - {name}: {why}")

    return results
