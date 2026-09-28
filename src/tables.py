"""表格抽取后端。

两个可选后端，产出同一种 `TableBlock`：

1. **pymupdf**（默认）
   - 用 `page.find_tables()`，基于框线 + 文本对齐
   - 快，一次遍历搞定
   - 需要 PyMuPDF >= 1.23

2. **pdfplumber**（备用，识别率不够时切换）
   - `page.extract_tables()`
   - 对"有框线但不规则"的财报表格通常更稳
   - 慢 5~10 倍，且需要把整份 PDF 打开一次预计算

3. **none**
   - 不抽表格，只抽文本。用于快速排查问题。

Windows 说明：这两个包都是纯 wheel，`pip install` 直接可用。
不用 camelot/tabula —— 那两个要 Java 或 Ghostscript，Windows 上很折腾。
"""

from __future__ import annotations

from typing import Optional

from .schema import TableBlock
from . import textnorm

BACKENDS = ("pymupdf", "pdfplumber", "none")


# --------------------------------------------------------------------------
# 后端 1: PyMuPDF
# --------------------------------------------------------------------------
def check_pymupdf_tables_available() -> tuple[bool, str]:
    from . import pymupdf_compat
    if not pymupdf_compat.is_available():
        return False, "未安装 PyMuPDF，请执行: pip install PyMuPDF"
    fitz = pymupdf_compat.fitz
    if not hasattr(fitz.Page, "find_tables"):
        return False, "PyMuPDF 版本过低（需 >= 1.23），请执行: pip install -U PyMuPDF"
    return True, "ok"


STRATEGIES = ("auto", "text", "lines", "lines_strict")


def _find_tables_raw(page, strategy: str):
    """调用 find_tables，兼容不支持 strategy 参数的老版本。"""
    try:
        return page.find_tables(strategy=strategy)
    except TypeError:
        return page.find_tables()


def _useful_cell_count(blocks: list[TableBlock]) -> int:
    """统计非空单元格数，用来判断哪个策略抽得更好。"""
    n = 0
    for b in blocks:
        for row in b.cells:
            for c in row:
                if textnorm.clean_cell(c):
                    n += 1
    return n


def extract_tables_pymupdf(page, page_no: int, section: str = "",
                           strategy: str = "auto") -> list[TableBlock]:
    """从 PyMuPDF 的 Page 对象抽表。

    strategy 两个取值各有各的死角，**必须都试**（所以默认 "auto"）：

      - "lines"：要求表格有完整网格线。对有边框的表很准，
        但中文财报普遍是"三线表"（只有上/中/下三条横线、没有竖线），
        "lines" 在平安年报上几乎抽不到东西
        （53 页抽样：1 张表 / 12 个单元格）。

      - "text"：按文字对齐推断列边界（需要 PyMuPDF >= 1.23.11 左右）。
        对三线表有效（同一份抽样：30 张表 / 7279 个单元格），
        但对完整网格线的表反而识别不出来。
    """
    if strategy != "auto":
        return _extract_with_strategy(page, page_no, section, strategy)

    best: list[TableBlock] = []
    best_score = -1
    for s in ("lines", "text"):
        got = _extract_with_strategy(page, page_no, section, s)
        score = _useful_cell_count(got)
        if score > best_score:
            best, best_score = got, score
    return best


def _extract_with_strategy(page, page_no: int, section: str,
                           strategy: str) -> list[TableBlock]:
    out: list[TableBlock] = []

    try:
        finder = _find_tables_raw(page, strategy)
    except Exception as exc:  # 某些 PDF 结构会直接抛错，不该让整份文档失败
        return [TableBlock(index=1, page=page_no, extractor="pymupdf",
                           section=section,
                           warnings=[f"find_tables({strategy}) 失败: {exc}"])]

    tables = getattr(finder, "tables", []) or []
    for i, tab in enumerate(tables, start=1):
        try:
            cells = tab.extract()
        except Exception as exc:
            out.append(TableBlock(index=i, page=page_no, extractor="pymupdf",
                                  section=section,
                                  warnings=[f"extract 失败: {exc}"]))
            continue

        header = []
        try:
            hdr = tab.header
            if hdr is not None and getattr(hdr, "names", None):
                header = [textnorm.clean_cell(h) for h in hdr.names]
        except Exception:
            header = []

        cells = textnorm.drop_empty_rows(cells)
        n_rows = len(cells)
        n_cols = max((len(r) for r in cells), default=0)

        warnings = textnorm.detect_merged_cells(cells)
        # 全空表直接丢掉
        if all(textnorm.clean_cell(c) == "" for r in cells for c in r):
            continue

        quality, ratio = textnorm.table_profile(cells)
        warnings += textnorm.table_quality_warnings(cells, quality)

        out.append(TableBlock(
            index=i,
            page=page_no,
            bbox=list(getattr(tab, "bbox", []) or []),
            n_rows=n_rows,
            n_cols=n_cols,
            cells=[[c for c in row] for row in cells],
            header=header,
            text=textnorm.linearize_table(cells, header or None),
            extractor="pymupdf",
            section=section,
            quality=quality,
            numeric_ratio=ratio,
            warnings=warnings,
        ))
    return out


# --------------------------------------------------------------------------
# 后端 2: pdfplumber
# --------------------------------------------------------------------------
def check_pdfplumber_available() -> tuple[bool, str]:
    try:
        import pdfplumber  # noqa: F401
        return True, "ok"
    except ImportError:
        return False, "未安装 pdfplumber，请执行: pip install pdfplumber"


def extract_all_tables_pdfplumber(pdf_path: str) -> dict[int, list[TableBlock]]:
    """一次性把整份 PDF 的表都抽出来，返回 {物理页号: [TableBlock]}。

    注意 pdfplumber 的 page 是 0-based，这里 +1 转成和 PyMuPDF 一致的 1-based。
    """
    import pdfplumber

    result: dict[int, list[TableBlock]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for pidx, page in enumerate(pdf.pages):
            page_no = pidx + 1
            blocks: list[TableBlock] = []

            # 用 find_tables() 而不是 extract_tables()：前者能拿到每张表的 bbox，
            # 抽取层要靠这个 bbox 去标记"落在表格区域里的文本块"，
            # 否则表格内容会同时以文本和表格两种形式进入索引，数字重复。
            found = []
            try:
                found = list(page.find_tables())
            except Exception:
                found = []

            if found:
                for i, tab in enumerate(found, start=1):
                    try:
                        cells = tab.extract()
                    except Exception as exc:
                        blocks.append(TableBlock(index=i, page=page_no,
                                                 extractor="pdfplumber",
                                                 warnings=[f"extract 失败: {exc}"]))
                        continue
                    if not cells:
                        continue
                    if all(textnorm.clean_cell(c) == "" for r in cells for c in r):
                        continue
                    header = []
                    try:
                        if getattr(tab, "header", None) and tab.header.names:
                            header = [textnorm.clean_cell(h) for h in tab.header.names]
                    except Exception:
                        header = []
                    cells = textnorm.drop_empty_rows(cells)
                    bbox = list(getattr(tab, "bbox", []) or [])
                    quality, ratio = textnorm.table_profile(cells)
                    blocks.append(TableBlock(
                        index=i,
                        page=page_no,
                        bbox=[float(v) for v in bbox] if len(bbox) == 4 else [],
                        n_rows=len(cells),
                        n_cols=max((len(r) for r in cells), default=0),
                        cells=[[c for c in row] for row in cells],
                        header=header,
                        text=textnorm.linearize_table(cells, header or None),
                        extractor="pdfplumber",
                        quality=quality,
                        numeric_ratio=ratio,
                        warnings=textnorm.detect_merged_cells(cells)
                        + textnorm.table_quality_warnings(cells, quality),
                    ))
            else:
                # 退路：老版本 pdfplumber 没有 find_tables，或者该页识别失败。
                # 没有 bbox，表格区域的文本会被重复收录一次，但至少不丢数据。
                try:
                    raw_tables = page.extract_tables() or []
                except Exception as exc:
                    result[page_no] = [TableBlock(index=1, page=page_no,
                                                  extractor="pdfplumber",
                                                  warnings=[f"extract_tables 失败: {exc}"])]
                    continue
                for i, cells in enumerate(raw_tables, start=1):
                    if not cells or all(textnorm.clean_cell(c) == "" for r in cells for c in r):
                        continue
                    cells = textnorm.drop_empty_rows(cells)
                    quality, ratio = textnorm.table_profile(cells)
                    blocks.append(TableBlock(
                        index=i, page=page_no,
                        n_rows=len(cells),
                        n_cols=max((len(r) for r in cells), default=0),
                        cells=[[c for c in row] for row in cells],
                        header=[],
                        text=textnorm.linearize_table(cells, None),
                        extractor="pdfplumber",
                        quality=quality,
                        numeric_ratio=ratio,
                        warnings=textnorm.detect_merged_cells(cells)
                        + textnorm.table_quality_warnings(cells, quality)
                        + ["缺少 bbox，表格区域文本可能被重复收录"],
                    ))
            result[page_no] = blocks
    return result


# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------
def get_table_extractor(backend: str):
    """返回一个可调用对象，签名见下面两个 wrapper。"""
    if backend not in BACKENDS:
        raise ValueError(f"未知表格后端 {backend!r}，可选：{BACKENDS}")
    if backend == "none":
        return None
    return {"pymupdf": extract_tables_pymupdf,
            "pdfplumber": extract_all_tables_pdfplumber}[backend]


def probe_backends() -> dict[str, tuple[bool, str]]:
    """检查各后端可用性，用于 CLI 的 `doctor` 子命令。"""
    ok_p, msg_p = check_pymupdf_tables_available()
    ok_pl, msg_pl = check_pdfplumber_available()
    return {"pymupdf": (ok_p, msg_p), "pdfplumber": (ok_pl, msg_pl)}
