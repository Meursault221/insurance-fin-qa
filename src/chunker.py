"""ExtractedDocument -> list[Chunk]。

切分策略（针对财报这种"章节 + 大表格"的文档专门设计）:

1. **表格原子化**：一张表永远切成一个完整 chunk，绝不从中间劈开。
   财报表格劈开 = 数字对不上号，检索回来也没法用。

2. **不跨页**：默认每个 chunk 只归属一页（page_end == page）。
   这样"页码"这个来源字段永远精确，不会出现"第 12-13 页"这种模糊引用。
   代价是段落可能被页边界截断，但换来的是引用可信 —— 对作业更划算。
   （想改成允许跨页，把 ChunkConfig.allow_cross_page 设为 True。）

3. **标题有条件开新块**：只有当缓冲里已经攒够 min_chars 时，标题才另起一块。
   年报里小标题极多，逢标题就断会产生一堆几十字的碎块，检索时全是噪声；
   让标题和它下面的正文待在一起，语义反而更完整。

4. **句末优先断句**：达到长度上限时，往回找最近的 。；！？ 或换行处切，
   而不是硬切在字中间。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .schema import Chunk, ExtractedDocument, PageRecord, TableBlock, TextBlock

# 断句优先级（从高到低）
_SENTENCE_ENDS = "。！？；\n"
_SOFT_ENDS = "，、）】"


@dataclass
class ChunkConfig:
    max_chars: int = 700            # 单 chunk 目标长度（中文字符）
    min_chars: int = 120            # 低于此长度的尾部碎块并入上一块
    overlap_chars: int = 100        # 同页相邻块之间的重叠，防边界丢信息
    include_table_text: bool = False  # 是否把表格区域内的原始文本行也算进来
    keep_tables: bool = True
    drop_short_chunks: bool = False
    allow_cross_page: bool = False
    max_table_chars: int = 6000     # 超过这个长度的表也强行保留（只告警不切）


def _build_prefix(company: str, report_name: str, page: int, section: str) -> str:
    parts = [f"【{company}·{report_name}】", f"第{page}页"]
    if section:
        parts.append(section)
    return " ".join(parts)


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """把一段长文本切成若干片，尽量在句末断。"""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    pieces: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            # 从 end 往回找最近的句末
            window_start = max(start + max_chars // 2, start + 1)
            cut = -1
            for i in range(end, window_start, -1):
                if text[i - 1] in _SENTENCE_ENDS:
                    cut = i
                    break
            if cut < 0:
                for i in range(end, window_start, -1):
                    if text[i - 1] in _SOFT_ENDS:
                        cut = i
                        break
            if cut > 0:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return pieces


def _ordered_items(page: PageRecord, cfg: ChunkConfig):
    """把一页的文本块和表格按纵坐标排成阅读顺序，产出 ("text"|"table", obj)。"""
    items: list[tuple[float, int, str, object]] = []
    for blk in page.blocks:
        if blk.kind == "page_footer":
            continue
        if blk.kind == "table_text" and not cfg.include_table_text:
            continue
        y = blk.bbox[1] if len(blk.bbox) >= 2 else 0.0
        items.append((y, 0, "text", blk))
    if cfg.keep_tables:
        for tab in page.tables:
            if tab.empty:
                continue
            y = tab.bbox[1] if len(tab.bbox) >= 2 else 0.0
            items.append((y, 1, "table", tab))
    items.sort(key=lambda t: (t[0], t[1]))
    for _y, _t, kind, obj in items:
        yield kind, obj


def chunk_document(doc: ExtractedDocument, cfg: Optional[ChunkConfig] = None) -> list[Chunk]:
    cfg = cfg or ChunkConfig()
    meta = doc.meta
    chunks: list[Chunk] = []
    seq = 0

    def emit(text: str, page: int, section: str, block_type: str,
             tables_payload: Optional[list[dict]] = None,
             printed_page: Optional[str] = None) -> None:
        nonlocal seq
        text = text.strip()
        if not text:
            return
        seq += 1
        chunks.append(Chunk(
            chunk_id=f"{meta.doc_id}_p{page:04d}_c{seq:04d}",
            doc_id=meta.doc_id,
            company=meta.company,
            report_year=meta.report_year,
            report_name=meta.report_name,
            page=page,
            page_end=page,
            printed_page=printed_page,
            section=section,
            text=text,
            block_type=block_type,
            chunk_index=seq,
            n_chars=len(text),
            stock_codes=list(meta.stock_codes),
            source_file=meta.source_file,
            tables=tables_payload or [],
            context_prefix=_build_prefix(meta.company, meta.report_name, page, section),
        ))

    for page in doc.pages:
        buf: list[str] = []
        buf_len = 0
        buf_section = ""
        buf_types: set[str] = set()      # 缓冲里混了哪些类型，用于定 block_type

        def flush():
            nonlocal buf, buf_len, buf_types
            if not buf:
                return
            raw = "\n".join(buf).strip()
            if raw:
                if len(buf_types) == 1:
                    btype = next(iter(buf_types))
                elif buf_types:
                    btype = "mixed"
                else:
                    btype = "paragraph"
                for piece in _split_text(raw, cfg.max_chars, cfg.overlap_chars):
                    if cfg.drop_short_chunks and len(piece) < cfg.min_chars:
                        continue
                    emit(piece, page.page, buf_section, btype,
                         printed_page=page.printed_page)
            buf = []
            buf_len = 0
            buf_types = set()

        for kind, obj in _ordered_items(page, cfg):
            if kind == "table":
                tab: TableBlock = obj  # type: ignore[assignment]
                flush()
                body = tab.text
                head = f"〔表格〕{tab.caption}" if tab.caption else "〔表格〕"
                emit(f"{head}\n{body}", page.page, tab.section or buf_section,
                     "table",
                     tables_payload=[{
                         "index": tab.index,
                         "n_rows": tab.n_rows,
                         "n_cols": tab.n_cols,
                         "header": tab.header,
                         "cells": tab.cells,
                         "caption": tab.caption,
                         "warnings": tab.warnings,
                     }],
                     printed_page=page.printed_page)
                if len(body) > cfg.max_table_chars:
                    # 只告警，不切 —— 宁可长，不可断
                    pass
                continue

            blk: TextBlock = obj  # type: ignore[assignment]

            if blk.kind == "heading":
                # 只有缓冲里已经攒够内容才开新块。
                # 否则让标题和它下面的正文待在一起 —— 年报里小标题极多，
                # 逢标题就断会产生一堆几十字的碎块，检索时全是噪声。
                if buf and buf_len >= cfg.min_chars:
                    flush()
                buf_section = blk.section or blk.text
                buf_types.add("heading")
                buf.append(blk.text)
                buf_len += len(blk.text)
                continue

            if not buf_section:
                buf_section = blk.section
            elif blk.section and blk.section != buf_section and not buf:
                buf_section = blk.section

            buf_types.add("paragraph")
            buf.append(blk.text)
            buf_len += len(blk.text)

            if buf_len >= cfg.max_chars:
                flush()

        flush()

    # 收尾：把过短的尾巴并入前一个 chunk（只在同页、同章节时合并）
    if cfg.min_chars > 0:
        merged: list[Chunk] = []
        for c in chunks:
            if (merged and c.n_chars < cfg.min_chars
                    and merged[-1].page == c.page
                    and merged[-1].section == c.section
                    and merged[-1].block_type == c.block_type):
                prev = merged[-1]
                prev.text = f"{prev.text}\n{c.text}"
                prev.n_chars = len(prev.text)
                prev.tables.extend(c.tables)
            else:
                merged.append(c)
        chunks = merged

    # 重编号，保证 chunk_id 连续
    for i, c in enumerate(chunks, start=1):
        c.chunk_index = i
        c.chunk_id = f"{meta.doc_id}_p{c.page:04d}_c{i:04d}"

    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict:
    if not chunks:
        return {"count": 0}
    lens = [c.n_chars for c in chunks]
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.block_type] = by_type.get(c.block_type, 0) + 1
    return {
        "count": len(chunks),
        "total_chars": sum(lens),
        "avg_chars": round(sum(lens) / len(lens), 1),
        "min_chars": min(lens),
        "max_chars": max(lens),
        "by_type": by_type,
        "pages_covered": len({c.page for c in chunks}),
    }


# --------------------------------------------------------------------------
# 落盘 / 读回（JSONL，一行一个 chunk，便于流式处理）
# --------------------------------------------------------------------------
def save_chunks(chunks: Iterable[Chunk], path) -> int:
    import json
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_chunks(path) -> list[Chunk]:
    import json
    from pathlib import Path
    p = Path(path)
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Chunk.from_dict(json.loads(line)))
            except json.JSONDecodeError as exc:
                # 最常见的误用是拿抽取结果 data/extracted/*.json 来读 ——
                # 那是多行缩进的 JSON，不是每行一条的 jsonl。给个能看懂的提示，
                # 而不是让 json 库的报错糊在脸上。
                head = line.lstrip()[:1]
                if head == "{":
                    raise ValueError(
                        f"{p} 第 {lineno} 行不是合法的 jsonl（每行必须是一条独立的 "
                        f"JSON）。如果你传的是 data/extracted/*.json，那是抽取结果、"
                        f"不是 chunk 文件；chunk 文件在 data/chunks/*.jsonl。"
                    ) from exc
                raise
    return out
