"""文本清洗与表格线性化。

红线：**只做空白/符号层面的规范化，绝不碰数字本身。**
    - 允许：全角空格 -> 半角、NBSP -> 空格、零宽字符删除、连续空格压缩
    - 禁止：删逗号千分位、改小数点、单位换算、把 "—" 当 0、补全空单元格
"""

from __future__ import annotations

import re
from typing import Optional

# 零宽 / 方向控制字符，PDF 抽取时经常混进来
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍‎‏﻿⁠"), None)

# 各种"看起来像空格"的字符
_SPACE_LIKE = {
    " ": " ",   # NBSP
    "　": " ",   # 全角空格
    " ": " ",   # en space
    " ": " ",   # em space
    " ": " ",   # thin space
    " ": " ",   # narrow nbsp
    "\t": " ",
    "\r": " ",
}

# 常见于财务报表的"零值"占位符。我们**不**把它们转成 0，
# 只在需要判断"是否为空"时用得上。
DASH_PLACEHOLDERS = {"—", "–", "-", "－", "‐", "不适用", "N/A", "n/a", "/"}

_NUM_LIKE = re.compile(r"^[\(\（]?\s*[-+]?\d[\d,\s]*(\.[\d]+)?\s*[\)\）]?%?$")


def normalize(text: str, collapse_spaces: bool = True) -> str:
    """基础清洗。保留换行（表格行结构要靠它）。"""
    if not text:
        return ""
    text = text.translate(_ZERO_WIDTH)
    for src, dst in _SPACE_LIKE.items():
        text = text.replace(src, dst)
    if collapse_spaces:
        # 行内多空格压成一个，但不动换行
        text = re.sub(r"[ ]{2,}", " ", text)
        # 去掉行尾空格
        text = re.sub(r"[ ]+\n", "\n", text)
    return text.strip()


def clean_cell(cell: Optional[str]) -> str:
    """单元格清洗。

    - None（PDF 里就是空的/被合并掉的）-> ""，**不是 "0"**
    - 内部换行 -> 空格（保留所有字符，只是不再换行）
    """
    if cell is None:
        return ""
    s = normalize(str(cell), collapse_spaces=False)
    s = s.replace("\n", " ")
    s = re.sub(r"[ ]{2,}", " ", s)
    return s.strip()


def is_blank_cell(s: str) -> bool:
    return s.strip() == ""


def is_number_like(s: str) -> bool:
    s = s.strip()
    return bool(s) and bool(_NUM_LIKE.match(s))


# --------------------------------------------------------------------------
# 字体编码完整性检查
# --------------------------------------------------------------------------
# 正常中英文年报会用到的字符区间。落在这些区间之外的字符，
# 基本只有一种来源：PDF 字体缺少 ToUnicode 映射表，抽取时吐出的是
# 字形编码而不是真正的 Unicode（平安年报第 129 页实测）。
_COMMON_RANGES = (
    (0x0020, 0x007E),   # ASCII 可打印
    (0x00A0, 0x00FF),   # Latin-1 补充（° × ÷ 等）
    (0x0370, 0x03FF),   # 希腊字母（保险业用 α β γ 表示风险系数）
    (0x2010, 0x203B),   # 常用标点（— – ‘ ’ “ ” •）
    (0x20AC, 0x20AC),   # €
    (0x2100, 0x21FF),   # 字母式符号（™ №）
    (0x2460, 0x24FF),   # 带圈数字
    (0x2200, 0x22FF),   # 数学运算符（∑ √ ≠ ≈）
    (0x2500, 0x257F),   # 制表符
    (0x25A0, 0x25FF),   # 几何图形（■ ● ◆）
    (0x2600, 0x27BF),   # 杂项符号（✓ ★）
    (0x3000, 0x303F),   # CJK 标点
    (0x4E00, 0x9FFF),   # CJK 统一汉字
    (0xFE30, 0xFE4F),   # CJK 兼容形式
    (0xFF00, 0xFFEF),   # 全角字符
)


def suspicious_char_ratio(text: str) -> float:
    """可疑字符（正常中英文之外）占比。

    用来发现"字体编码缺失"的页面 —— 这类页面抽出来的字是乱码，
    即使不报错也会污染索引，必须显式标记出来提示人工复核/OCR。
    """
    s = "".join(ch for ch in (text or "") if not ch.isspace())
    if not s:
        return 0.0
    bad = sum(1 for ch in s
              if not any(lo <= ord(ch) <= hi for lo, hi in _COMMON_RANGES))
    return bad / len(s)


def strip_trailing_page_number(line: str) -> str:
    """去掉页脚那种"第 12 页 / 12"结尾，避免污染正文。"""
    s = re.sub(r"\s*第\s*\d+\s*页\s*(共\s*\d+\s*页)?\s*$", "", line)
    s = re.sub(r"\s*\d{1,3}\s*$", "", s) if len(s.strip()) > 12 else s
    return s.strip()


_PAGE_FOOTER_RE = re.compile(
    r"^(?:第\s*)?(\d{1,4})\s*(?:页)?(?:\s*[/／]\s*(?:共\s*)?(\d{1,4})\s*页?)?$"
)


def detect_printed_page_number(line: str) -> Optional[str]:
    """识别页脚印刷页码，如 "12" / "第 12 页" / "12 / 350"。

    返回原始字符串（不做 int 转换，因为有些是罗马数字）。
    只在整行非常短的时候才认为是页码，避免把正文里的数字误判。
    """
    s = line.strip()
    if not s or len(s) > 20:
        return None
    m = _PAGE_FOOTER_RE.match(s)
    if m:
        return m.group(1)
    if re.fullmatch(r"[ivxlcdmIVXLCDM]{1,7}", s):
        return s
    return None


# --------------------------------------------------------------------------
# 表格线性化
# --------------------------------------------------------------------------
def escape_md_cell(s: str) -> str:
    """管道表里 | 会破坏结构，转义掉。其余字符一律不动。"""
    return s.replace("|", "\\|")


def linearize_table(cells: list[list[Optional[str]]],
                    header: Optional[list[str]] = None) -> str:
    """把二维单元格转成 markdown 管道表。

    保留行列结构，保留每个单元格的原始字符（含千分位逗号、负号、括号负数）。
    空单元格写成空字符串，**不填 0**。
    """
    if not cells:
        return ""

    cleaned = [[clean_cell(c) for c in row] for row in cells]
    n_cols = max((len(r) for r in cleaned), default=0)
    if n_cols == 0:
        return ""

    # 补齐每行长度（用空串，不是 0）
    for row in cleaned:
        while len(row) < n_cols:
            row.append("")

    lines = []
    if header:
        head = [clean_cell(h) for h in header]
        while len(head) < n_cols:
            head.append("")
        head = head[:n_cols]
        lines.append("| " + " | ".join(escape_md_cell(h) for h in head) + " |")
        lines.append("| " + " | ".join(["---"] * n_cols) + " |")
        body = cleaned
        # extract() 出来的 cells 通常**包含**表头行，直接用会和上面那行重复。
        # 只在内容确实一致时才去掉，避免误删真正的数据行。
        if cleaned and cleaned[0] == head:
            body = cleaned[1:]
    else:
        # 没有表头信息：第一行当表头（财报表格第一行基本就是表头）
        lines.append("| " + " | ".join(escape_md_cell(c) for c in cleaned[0]) + " |")
        lines.append("| " + " | ".join(["---"] * n_cols) + " |")
        body = cleaned[1:]

    for row in body:
        lines.append("| " + " | ".join(escape_md_cell(c) for c in row) + " |")

    return "\n".join(lines)


def table_to_records(cells: list[list[Optional[str]]],
                     header: Optional[list[str]] = None) -> list[dict[str, str]]:
    """把表格转成 [{表头: 单元格}] 形式，方便第二阶段精确取数。

    表头有重名时自动加后缀 _2 / _3，不会丢列。
    """
    if not cells:
        return []

    cleaned = [[clean_cell(c) for c in row] for row in cells]
    if header:
        head = [clean_cell(h) for h in header]
        body = cleaned
        if cleaned and cleaned[0] == head:
            body = cleaned[1:]
    else:
        if len(cleaned) < 2:
            return []
        head = cleaned[0]
        body = cleaned[1:]

    seen: dict[str, int] = {}
    uniq_head: list[str] = []
    for i, h in enumerate(head):
        name = h or f"col{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        uniq_head.append(name)

    records = []
    for row in body:
        rec = {}
        for i, name in enumerate(uniq_head):
            rec[name] = row[i] if i < len(row) else ""
        records.append(rec)
    return records


def drop_empty_rows(cells: list[list[Optional[str]]]) -> list[list[Optional[str]]]:
    """去掉整行都为空的表格行。

    PyMuPDF 的 "text" 策略会把表格区域里每一行文字都当成一行，
    期间夹进大量全空行 —— 平安年报实测 9459 行里有 4538 行（48%）是全空的。
    这些行不含任何数据，留着只会让 chunk 变长、检索变糊。

    只删"整行全空"，任何一个单元格有内容就保留 —— 不会丢数据。
    """
    if not cells:
        return cells
    kept = [row for row in cells
            if any(clean_cell(c) != "" for c in row)]
    return kept or cells


NUMERIC_TABLE_RATIO = 0.2
"""表格里"数字型单元格"占比超过这个值，就认为它是以数据为主的表。

财报里的表格分两类，检索时应当区别对待：
  - 数字型（资产负债表、保费收入表…）：必须保留二维结构，取数要靠 cells
  - 文字型（董事简历、关键审计事项、会计政策…）：本质是带标签的段落

平安年报实测：182 张表里有 62 张数字占比 < 0.15，即三分之一是文字型。
"""

PROSE_TABLE_AVG_LEN = 25
"""文字型表格的平均单元格长度超过这个值，就怀疑它根本不是表格。"""


def table_profile(cells: list[list[Optional[str]]]) -> tuple[str, float]:
    """给表格打质量标签，返回 (label, 数字占比)。

    label ∈ {"numeric", "text", "empty"}。

    **只做标记，不删任何数据。** 因为纯文字的真表格（董事简历、
    关键审计事项）和图文排版的误检（董事长致辞的两栏正文）在
    "数字占比"上完全一样，都接近 0 —— 靠阈值区分不开，
    贸然删除会误杀真表格。所以这里只把判断依据记下来，
    交给第二阶段决定怎么用。
    """
    vals = [clean_cell(c) for r in (cells or []) for c in r if clean_cell(c)]
    if not vals:
        return "empty", 0.0
    ratio = sum(1 for v in vals if is_number_like(v)) / len(vals)
    label = "numeric" if ratio >= NUMERIC_TABLE_RATIO else "text"
    return label, round(ratio, 4)


def table_quality_warnings(cells: list[list[Optional[str]]],
                           label: str) -> list[str]:
    """对"可疑的表格"给出人工复核提示。

    只提示，不拦截 —— 误检的代价是检索质量下降，
    误删的代价是数据丢失，后者严重得多。
    """
    vals = [clean_cell(c) for r in (cells or []) for c in r if clean_cell(c)]
    if not vals or label != "text":
        return []
    avg = sum(len(v) for v in vals) / len(vals)
    if avg >= PROSE_TABLE_AVG_LEN:
        return [f"疑似图文排版区域被误识别为表格"
                f"（无数字、单元格平均 {avg:.0f} 字），建议人工复核"]
    return []


def detect_merged_cells(cells: list[list[Optional[str]]]) -> list[str]:
    """粗判是否存在合并单元格（用 None 密度 + 行宽不齐估计），返回告警文案。"""
    warns: list[str] = []
    if not cells:
        return warns
    widths = {len(r) for r in cells}
    if len(widths) > 1:
        warns.append(f"各行单元格数不一致({sorted(widths)})，可能存在合并单元格")
    none_cnt = sum(1 for r in cells for c in r if c is None)
    total = sum(len(r) for r in cells)
    if total and none_cnt / total > 0.3:
        warns.append(f"{none_cnt}/{total} 个单元格为空值(None)，可能是合并单元格或跨页表格")
    return warns
