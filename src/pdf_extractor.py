"""PDF -> ExtractedDocument。

流程（逐页）:
    1. PyMuPDF `get_text("dict")` 拿到带坐标/字号/字体的 line
    2. line 合并成 block（段落 / 标题）
    3. 抽表格（PyMuPDF 或 pdfplumber 后端）
    4. 标题识别 + 章节路径追踪
    5. 标记落在表格区域内的文本块（保留，不丢）
    6. 识别页脚印刷页码
    7. 组装成 PageRecord / ExtractedDocument

绝不修改数字。绝不编造页码。
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
from pathlib import Path
from typing import Any, Optional

from . import config, pymupdf_compat, sections, tables as tables_mod, textnorm
from .schema import (DocumentMeta, ExtractedDocument, PageRecord, TableBlock,
                     TextBlock)

SUSPICIOUS_CHAR_RATIO = 0.15
"""正文里"非正常中英文字符"超过这个比例，就判定该页字体编码有问题。

阈值取 15% 是为了容忍正常年报里的数学符号、项目符号（■ ● ✓ β 等）——
它们零星出现，占比通常远低于 1%。
"""

# 判定"加粗"的字体名关键字（中文年报常用 黑体/中黑/Bold/Heavy）
BOLD_FONT_HINTS = ("bold", "heavy", "black", "semibold", "demibold",
                   "hei", "黑体", "中黑", "粗")

# PyMuPDF span flag 第 4 位(值 16) 表示 bold
_BOLD_FLAG = 1 << 4

# 相邻 line 合并成段落的垂直容差（相对于行高）
_LINE_MERGE_GAP_RATIO = 0.9
_LINE_MERGE_GAP_MIN = 5.0

# 一页文字少于这个数就怀疑是扫描件
_SCANNED_PAGE_CHAR_THRESHOLD = 40


# ==========================================================================
# 基础工具
# ==========================================================================
def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _is_bold(font_name: str, flags: int) -> bool:
    if flags & _BOLD_FLAG:
        return True
    low = (font_name or "").lower()
    return any(h in low for h in BOLD_FONT_HINTS)


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def build_doc_id(company_key: str, report_year: int, report_type: str) -> str:
    suffix = {"annual": "annual", "semi_annual": "semi"}.get(report_type, report_type)
    return f"{company_key}_{report_year}_{suffix}"


def build_meta(
    pdf_path: str | Path,
    company: str,
    company_key: str,
    report_year: int,
    report_name: str,
    *,
    company_full: str = "",
    stock_codes: Optional[list[str]] = None,
    exchange: str = "",
    report_type: str = "annual",
    source_url: str = "",
    with_hash: bool = True,
) -> DocumentMeta:
    """构造文档元数据。company / report_year / report_name 最终会出现在每条来源里。"""
    p = Path(pdf_path)
    try:
        rel = str(p.resolve().relative_to(config.PROJECT_ROOT))
    except Exception:
        rel = str(p)

    return DocumentMeta(
        doc_id=build_doc_id(company_key, report_year, report_type),
        company=company,
        company_full=company_full,
        company_key=company_key,
        stock_codes=list(stock_codes or []),
        exchange=exchange,
        report_year=report_year,
        report_name=report_name,
        report_type=report_type,
        source_file=p.name,
        source_path=rel,
        source_url=source_url,
        file_sha256=sha256_file(p) if with_hash else "",
        fetched_at=_now_iso(),
    )


# ==========================================================================
# line -> block
# ==========================================================================
class _Line:
    __slots__ = ("page", "block_no", "text", "bbox", "size", "font", "bold")

    def __init__(self, page, block_no, text, bbox, size, font, bold):
        self.page = page
        self.block_no = block_no
        self.text = text
        self.bbox = bbox
        self.size = size
        self.font = font
        self.bold = bold

    @property
    def height(self) -> float:
        return max(self.bbox[3] - self.bbox[1], 1.0)


_GROUPED_NUM_TAIL = re.compile(r"\d,\d{3}$")
_GROUPED_NUM_HEAD = re.compile(r"\d,\d{3}")


def _join_spans(spans) -> str:
    """拼接同一行内的 span。

    不能简单地 "" 拼接：PyMuPDF 会把相邻两个财务数字拆成两个 span，
    直接拼就把 "376,112 1,304,712" 粘成 "376,1121,304,712" ——
    变成一个根本不存在的数，取数时无法还原（合并股东权益变动表实测 10 处）。

    只在"千分位分组数字挨着千分位分组数字"的边界补一个空格，
    其余边界原样保留，避免把被拆开的普通词语拆出多余空格。
    """
    out = ""
    for s in spans:
        t = s.get("text", "")
        if out and t:
            glue = ((_GROUPED_NUM_TAIL.search(out) and t[0].isdigit())
                    or (out[-1].isdigit() and _GROUPED_NUM_HEAD.match(t)))
            if glue:
                out += " "
        out += t
    return out


def _page_lines(page, page_no: int) -> list[_Line]:
    """把一页里的所有文本行按阅读顺序取出来。"""
    try:
        d = page.get_text("dict", sort=True)
    except TypeError:
        # 老版本 PyMuPDF 不支持 sort 参数
        d = page.get_text("dict")

    lines: list[_Line] = []
    for b_idx, block in enumerate(d.get("blocks", [])):
        if block.get("type") != 0:      # 1 = 图片
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = _join_spans(spans)
            if not text.strip():
                continue
            sizes = [float(s.get("size", 0.0)) for s in spans]
            fonts = [s.get("font", "") or "" for s in spans]
            flags = [int(s.get("flags", 0)) for s in spans]
            bold = any(_is_bold(f, fl) for f, fl in zip(fonts, flags))
            lines.append(_Line(
                page=page_no,
                block_no=block.get("number", b_idx),
                text=textnorm.normalize(text, collapse_spaces=False),
                bbox=list(line.get("bbox", [0, 0, 0, 0])),
                size=max(sizes) if sizes else 0.0,
                font=max(fonts, key=len) if fonts else "",
                bold=bold,
            ))
    return lines


def _merge_lines_into_blocks(lines: list[_Line],
                             body_font_size: float) -> list[tuple[str, int, _Line]]:
    """把 line 合并成 block。返回 [(text, level, first_line_of_block)]。"""
    results: list[tuple[str, int, _Line]] = []
    cur: list[_Line] = []

    def flush():
        if not cur:
            return
        text = "\n".join(l.text for l in cur).strip()
        if text:
            kind, level = sections.classify_block(
                cur[0].text, cur[0].size, body_font_size, cur[0].bold)
            results.append((text, level if kind == "heading" else 0, cur[0]))
        cur.clear()

    for ln in lines:
        # 先判断这一行自己是不是标题 —— 是标题就先把累积的段落吐出去
        kind, level = sections.classify_block(
            ln.text, ln.size, body_font_size, ln.bold)

        if kind == "heading":
            flush()
            results.append((ln.text.strip(), level, ln))
            continue

        if not cur:
            cur.append(ln)
            continue

        prev = cur[-1]
        same_block = prev.block_no == ln.block_no
        same_size = abs(prev.size - ln.size) < 0.6
        gap = ln.bbox[1] - prev.bbox[3]
        gap_ok = gap <= max(_LINE_MERGE_GAP_RATIO * prev.height, _LINE_MERGE_GAP_MIN)

        if same_block and same_size and gap_ok:
            cur.append(ln)
        else:
            flush()
            cur.append(ln)

    flush()
    return results


def _infer_body_font_size(lines: list[_Line]) -> float:
    """按字符数加权取众数，得到正文字号。"""
    weights: dict[float, int] = {}
    for ln in lines:
        key = round(ln.size, 1)
        weights[key] = weights.get(key, 0) + len(ln.text.strip())
    if not weights:
        return 0.0
    return max(weights.items(), key=lambda kv: kv[1])[0]


# ==========================================================================
# 表格相关
# ==========================================================================
def _bbox_center_inside(bbox: list[float], outer: list[float]) -> bool:
    if len(bbox) < 4 or len(outer) < 4:
        return False
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


# 纯标点/空白，不配当表格标题
_PUNCT_ONLY = re.compile(r"^[\s。，、；：！？（）()【】\[\]·—\-–_/\\|]+$")


_DIGIT_RUN = re.compile(r"\d[\d,]*(?:\.\d+)?")
_COVER_STRIP = re.compile(r"[\s|]+")


def _content_captured(text: str, tab: TableBlock) -> bool:
    """判断文本块的内容是否真的被这张表收录了。

    不判断就会丢数据：PyMuPDF 的 text 策略经常把一大片区域认成"一张表"，
    却没把区域内的数字抽进单元格。这些文本块一旦被标成 table_text，
    就会被 chunker 跳过，数字永远进不了索引
    （平安年报第 57 页实测丢了 18,536 / 6,082 / 139,425 等真实保费数据）。

    判据以数字为准 —— 数字是财务报表的命根子。块里每个数字串都能在
    表格里找到才算收录；只要有一个对不上，就把这块保留成普通正文。
    宁可重复，不可丢失。
    """
    want = set(_DIGIT_RUN.findall(text or ""))
    if not want:
        return _COVER_STRIP.sub("", text or "") in _COVER_STRIP.sub("", tab.text or "")
    have = set(_DIGIT_RUN.findall(tab.text or ""))
    for row in tab.cells:
        for c in row:
            have |= set(_DIGIT_RUN.findall(c or ""))
    return want <= have


def _section_at_y(blocks: list[TextBlock], y: float) -> str:
    """取纵坐标 y 之上、最近的那个文本块所属的章节。

    用来给表格定章节 —— 不能用"整页跑完后的 tracker 状态"，
    那会拿到页面最后一节的标题，跟表格实际位置对不上。
    """
    best, best_y = "", -1e18
    for b in blocks:
        if len(b.bbox) < 2 or b.kind == "table_text":
            continue
        by = b.bbox[1]
        if by <= y + 1.0 and by >= best_y:
            best_y, best = by, b.section
    return best


def _find_table_caption(blocks: list[TextBlock], tab: TableBlock) -> str:
    """找表格上方最近的一小块文字，当作"单位：xxx"这类注释。"""
    if len(tab.bbox) < 4:
        return ""
    best, best_dy = "", 1e9
    for b in blocks:
        txt = (b.text or "").strip()
        if not txt or len(txt) < 2:
            continue
        if _PUNCT_ONLY.match(txt):      # 排除孤零零的 "。" 这类噪声
            continue
        if b.inside_table:              # 表格自己的内容不算标题
            continue
        # 只认真正像"表格题注"的文字：含"单位"，或者本身很短。
        # 否则会把表格上方的一整段正文误当成标题。
        if "单位" not in txt and len(txt) > 25:
            continue
        if len(b.bbox) < 4:
            continue
        dy = tab.bbox[1] - b.bbox[3]
        if 0 <= dy < 40 and dy < best_dy:
            best, best_dy = txt, dy
    return best


# 页眉/页脚所在的边距带，占页高的比例（上下各取这么多）
_RUNNING_BAND_RATIO = 0.12
# 同一段文字至少出现在这么多页，才认定是页眉/页脚
_RUNNING_MIN_PAGES = 4
# 页眉里通常带着页码（"……公司15"），比对前先把数字抹掉
_DIGITS = re.compile(r"\d+")


def _mark_running_headers(pages: list[PageRecord]) -> int:
    """把每页重复出现在上下边距里的文字标记成 page_footer，返回标记数量。

    为什么必须做：年报每页顶部/底部都印着
    "中国平安保险（集团）股份有限公司 15" 这样的书眉。
    它不是章节标题，但字号偏大、还常加粗，会被字号规则判成标题，
    于是污染 section 字段 —— 平安年报实测 26.7% 的 chunk 中招。

    判据是"位置在边距内" + "去掉数字后同一文本重复出现 >= 4 页"。
    真标题不会在 4 页以上完全相同地重复，所以不会误杀。
    """
    from collections import defaultdict

    groups: dict[str, list[TextBlock]] = defaultdict(list)
    for page in pages:
        if not page.height:
            continue
        top = page.height * _RUNNING_BAND_RATIO
        bottom = page.height * (1 - _RUNNING_BAND_RATIO)
        for blk in page.blocks:
            if blk.kind in ("table_text", "page_footer"):
                continue
            if len(blk.bbox) < 4:
                continue
            y0, y1 = blk.bbox[1], blk.bbox[3]
            if not (y1 <= top or y0 >= bottom):
                continue
            # 书眉永远是单独一行。这一条很关键：财务报表附注每页顶部都有
            # "2025年度\n（除特别注明外，金额单位为人民币百万元）" 这样的
            # 表格题注 + 单位说明，它同样"重复出现在多页的边距里"，
            # 但它是正文，而且是解读数字的关键（丢了就不知道单位是百万还是千）。
            if "\n" in blk.text or len(blk.text) > 60:
                continue
            key = _DIGITS.sub("", blk.text).strip()
            if 4 <= len(key) <= 40:
                groups[key].append(blk)

    marked = 0
    for blks in groups.values():
        if len(blks) >= _RUNNING_MIN_PAGES:
            for blk in blks:
                blk.kind = "page_footer"
                blk.level = 0
                marked += 1
    return marked


def _assign_sections(pages: list[PageRecord]) -> None:
    """全篇重算章节路径。

    必须等 _mark_running_headers 跑完再做 —— 页眉没剔掉之前算出来的
    章节路径是错的。只有 heading 会推动标题栈，页脚/表格区文本不参与。
    """
    tracker = sections.SectionTracker()
    for page in pages:
        for blk in page.blocks:
            if blk.kind == "heading":
                blk.section = tracker.update("heading", blk.level, blk.text)
            else:
                blk.section = tracker.current
        for tab in page.tables:
            y = tab.bbox[1] if len(tab.bbox) >= 2 else 0.0
            tab.section = _section_at_y(page.blocks, y) or tracker.current


# ==========================================================================
# 主流程
# ==========================================================================
def extract_pdf(
    pdf_path: str | Path,
    meta: DocumentMeta,
    *,
    table_backend: str = "pymupdf",
    table_strategy: str = "auto",
    detect_printed_page: bool = True,
    min_chars_per_page: int = 0,
) -> ExtractedDocument:
    """抽一份 PDF。

    Args:
        table_backend: "pymupdf" | "pdfplumber" | "none"
        detect_printed_page: 是否尝试识别页脚印刷页码。

    说明：落在表格区域里的原始文本行**永远保留**在 blocks 里
    （kind="table_text"），是否收录由 chunker 的 include_table_text 决定。
    抽取层不做任何丢弃，保证 JSON 是完整的原始证据。

    Returns:
        ExtractedDocument
    """
    fitz = pymupdf_compat.require()

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在：{pdf_path}")

    warnings: list[str] = []

    # --- pdfplumber 后端需要预先扫全量 ---
    precomputed: dict[int, list[TableBlock]] = {}
    if table_backend == "pdfplumber":
        ok, msg = tables_mod.check_pdfplumber_available()
        if not ok:
            warnings.append(f"{msg}，已自动回退到 pymupdf")
            table_backend = "pymupdf"
        else:
            precomputed = tables_mod.extract_all_tables_pdfplumber(str(pdf_path))
    elif table_backend == "pymupdf":
        ok, msg = tables_mod.check_pymupdf_tables_available()
        if not ok:
            warnings.append(f"{msg}，本次不抽表格（可用 pdfplumber 后端重试）")
            table_backend = "none"

    doc = fitz.open(str(pdf_path))
    pages: list[PageRecord] = []

    total_blocks = total_tables = total_chars = total_cells = 0
    scanned_pages: list[int] = []
    garbled_pages: list[int] = []
    quality_counts: dict[str, int] = {}
    uncaptured_blocks = 0

    try:
        # 先算出全篇正文字号（用前 30 页做样本，足够稳定，也省时间）
        sample_lines: list[_Line] = []
        for i in range(min(doc.page_count, 30)):
            sample_lines.extend(_page_lines(doc[i], i + 1))
        body_font_size = _infer_body_font_size(sample_lines)
        del sample_lines

        for pidx in range(doc.page_count):
            page_no = pidx + 1
            page = doc[pidx]
            rect = page.rect

            lines = _page_lines(page, page_no)
            raw_chars = sum(len(l.text.strip()) for l in lines)
            if raw_chars < _SCANNED_PAGE_CHAR_THRESHOLD:
                img_count = len(page.get_images(full=True))
                if img_count > 0:
                    scanned_pages.append(page_no)

            # --- 表格 ---
            if table_backend == "pdfplumber":
                page_tables = precomputed.get(page_no, [])
            elif table_backend == "pymupdf":
                page_tables = tables_mod.extract_tables_pymupdf(
                    page, page_no, strategy=table_strategy)
            else:
                page_tables = []

            # --- 文本块 ---
            page_blocks: list[TextBlock] = []
            merged = _merge_lines_into_blocks(lines, body_font_size)
            for idx, (text, level, first_line) in enumerate(merged, start=1):
                kind = "heading" if level > 0 else "paragraph"
                # 章节路径先留空：页眉的剔除和章节重算要等全篇扫完才能做，
                # 见 _mark_running_headers / _assign_sections。
                page_blocks.append(TextBlock(
                    index=idx,
                    page=page_no,
                    bbox=list(first_line.bbox),
                    text=text,
                    kind=kind,
                    level=level,
                    section="",
                    font_size=round(first_line.size, 2),
                    font_name=first_line.font,
                    bold=first_line.bold,
                ))

            # --- 标记落在表格区域里的文本块 ---
            # 这些块照旧留在 JSON 里（一个字符都不丢），只是打上 kind="table_text"，
            # 由 chunker 决定要不要收录，避免同一批数字出现两遍。
            for tab in page_tables:
                for blk in page_blocks:
                    if _bbox_center_inside(blk.bbox, tab.bbox):
                        blk.inside_table = tab.index
                        if _content_captured(blk.text, tab):
                            blk.kind = "table_text"
                        else:
                            # 表格没抽到的内容 —— 保留成普通正文。
                            # 如果照样标 table_text，chunker 会跳过它，数字就丢了。
                            blk.kind = "paragraph"
                            uncaptured_blocks += 1

            # --- 表格题注（章节归属统一在 _assign_sections 里算）---
            for tab in page_tables:
                cap = _find_table_caption(page_blocks, tab)
                if cap:
                    tab.caption = cap

            # --- 页脚印刷页码 ---
            printed = None
            if detect_printed_page:
                for blk in reversed(page_blocks[-3:] if len(page_blocks) >= 3 else page_blocks):
                    cand = textnorm.detect_printed_page_number(blk.text)
                    if cand is not None:
                        printed = cand
                        blk.printed_page = cand
                        # 页脚不是正文，打上标记让 chunker 跳过，
                        # 否则页码会被拼进正文尾部污染检索
                        blk.kind = "page_footer"
                        break

            # page.text 是给人看的"正文"速览：剔除表格区域和页脚，
            # 免得同一批数字出现两遍、页码混进正文。
            # 它们一个字都没丢 —— 都在 blocks(kind=table_text / page_footer) 和 tables 里。
            page_text = "\n".join(
                b.text for b in page_blocks
                if b.kind not in ("table_text", "page_footer"))

            # 字体编码完整性：有些 PDF 的字体没有 ToUnicode 表，
            # 抽出来的是字形编码而不是真 Unicode（平安年报第 129 页就是）。
            # 这种页不报错、但内容是乱码，必须显式标出来，否则会悄悄污染索引。
            page_warns: list[str] = []
            # 用"全页内容"而不是 page_text 来判定 —— 后者排除了表格区文本，
            # 而乱码恰恰经常整块落在表格区域里（平安年报第 129 页就是）。
            all_text = page_text + "\n" + "\n".join(
                b.text for b in page_blocks if b.kind == "table_text")
            all_text += "\n" + "\n".join(t.text for t in page_tables)
            ratio_bad = textnorm.suspicious_char_ratio(all_text)
            if ratio_bad >= SUSPICIOUS_CHAR_RATIO:
                page_warns.append(
                    f"疑似字体编码缺失，{ratio_bad:.0%} 的字符不是正常中英文"
                    f"（抽出内容为乱码，需 OCR 或换解析器）"
                )
                garbled_pages.append(page_no)

            rec = PageRecord(
                page=page_no,
                width=round(float(rect.width), 2),
                height=round(float(rect.height), 2),
                printed_page=printed,
                blocks=page_blocks,
                tables=page_tables,
                text=page_text,
                char_count=len(page_text),
                warnings=page_warns,
            )

            pages.append(rec)
            total_blocks += len(page_blocks)
            total_tables += len(page_tables)
            total_chars += rec.char_count
            total_cells += sum(t.n_rows * t.n_cols for t in page_tables)
            for t in page_tables:
                quality_counts[t.quality] = quality_counts.get(t.quality, 0) + 1
    finally:
        doc.close()

    # 全篇扫完之后才能做的两件事，顺序不能反：
    # 先剔页眉，再算章节（否则页眉会顶着标题栈污染 section）。
    running_marked = _mark_running_headers(pages)
    _assign_sections(pages)

    if scanned_pages:
        warnings.append(
            f"以下页面文字极少但含图片，疑似扫描件/图片版，需要 OCR：{scanned_pages[:20]}"
            + (" ..." if len(scanned_pages) > 20 else "")
        )

    if garbled_pages:
        warnings.append(
            f"以下页面疑似字体编码缺失（抽出乱码），索引质量受影响，建议 OCR："
            f"{garbled_pages[:20]}" + (" ..." if len(garbled_pages) > 20 else "")
        )

    n_text_tables = quality_counts.get("text", 0)
    if n_text_tables:
        warnings.append(
            f"{n_text_tables}/{total_tables} 张表是文字型（数字占比 < "
            f"{textnorm.NUMERIC_TABLE_RATIO:.0%}），其中一部分可能是图文排版被误识别；"
            f"数据未丢失，但检索质量会受影响，详见每张表的 quality 字段与 warnings"
        )

    meta.page_count = len(pages)

    extractor_version = pymupdf_compat.version()

    stats: dict[str, Any] = {
        "page_count": len(pages),
        "block_count": total_blocks,
        "table_count": total_tables,
        "table_cell_count": total_cells,
        "char_count": total_chars,
        "avg_chars_per_page": round(total_chars / len(pages), 1) if pages else 0,
        "scanned_page_count": len(scanned_pages),
        "scanned_pages": scanned_pages[:50],
        "garbled_page_count": len(garbled_pages),
        "garbled_pages": garbled_pages[:50],
        "running_header_blocks": running_marked,
        "uncaptured_table_blocks": uncaptured_blocks,
        "table_quality_counts": quality_counts,
        "body_font_size": body_font_size,
    }

    return ExtractedDocument(
        meta=meta,
        pages=pages,
        extractor="PyMuPDF",
        extractor_version=extractor_version,
        table_backend=table_backend,
        extracted_at=_now_iso(),
        body_font_size=body_font_size,
        stats={**stats, "warnings": warnings},
    )


# ==========================================================================
# 落盘 / 读回
# ==========================================================================
def save_extracted(doc: ExtractedDocument, out_dir: Optional[Path] = None) -> Path:
    import json
    out_dir = Path(out_dir) if out_dir else config.EXTRACTED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{doc.meta.doc_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)
    return path


def load_extracted(path: str | Path) -> ExtractedDocument:
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ExtractedDocument.from_dict(data)


def _fmt_quality(counts: dict[str, int]) -> str:
    """把 {numeric: 120, text: 62} 渲染成 "数字型 120 / 文字型 62"。"""
    name = {"numeric": "数字型", "text": "文字型", "empty": "空表",
            "unknown": "未判定"}
    if not counts:
        return "（未统计）"
    return " / ".join(f"{name.get(k, k)} {v}" for k, v in
                      sorted(counts.items(), key=lambda kv: -kv[1]))


def quick_report(doc: ExtractedDocument) -> str:
    """给命令行看的一页摘要。"""
    m = doc.meta
    s = doc.stats
    lines = [
        "=" * 66,
        f"文档      : {m.company} / {m.report_name}",
        f"doc_id    : {m.doc_id}",
        f"来源文件  : {m.source_file}",
        f"页数      : {m.page_count}",
        f"文本块    : {s.get('block_count')}",
        f"表格      : {s.get('table_count')}（{s.get('table_cell_count')} 个单元格）",
        f"  表格构成: {_fmt_quality(s.get('table_quality_counts') or {})}",
        f"总字符    : {s.get('char_count')}（平均 {s.get('avg_chars_per_page')}/页）",
        f"正文字号  : {s.get('body_font_size')}",
        f"解析器    : {doc.extractor} {doc.extractor_version} / 表格后端={doc.table_backend}",
        "=" * 66,
    ]
    for w in s.get("warnings", []) or []:
        lines.append(f"⚠️  {w}")
    return "\n".join(lines)
