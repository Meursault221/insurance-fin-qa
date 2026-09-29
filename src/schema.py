"""项目统一数据结构。

设计原则（很重要，直接关系到作业能不能拿分）:

1. **来源可回溯**：任何一个最终送给用户看的 `Chunk`，都必须自带
   (company, report_year, report_name, page, section) 五个字段，
   不需要再去别处查表。作业要求回答里显示"公司/报告/页码"，靠的就是这个。

2. **数字只搬运，不加工**：抽取过程中不做千分位清洗、不做单位换算、
   不做四舍五入、不把 "—" 或空单元格补成 0。PDF 里是什么就是什么。
   任何数字层面的"美化"都会让财报数据失真。

3. **表格双形态**：同时保留 `cells`(二维数组，真正的行列结构) 和
   `text`(线性化后的 markdown 管道表)。前者用于精确取数，
   后者用于喂给检索/大模型。

4. **页码是 1-based 的物理页序号**，即 PDF 阅读器里显示的页。
   注意：很多年报正文的"页脚印刷页码"和物理页序号不一致，
   我们两者都留字段，检索时用物理页序号（可复现），
   展示时如果印刷页码存在就一并显示。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional


# --------------------------------------------------------------------------
# 基础工具：让嵌套 dataclass 能方便地在 dict <-> object 之间转换
# --------------------------------------------------------------------------
def _from_dict(cls, data: dict[str, Any]):
    """把 dict 还原成 dataclass，忽略多余字段，自动递归嵌套 dataclass。"""
    if data is None:
        return None
    if not dataclasses.is_dataclass(cls):
        return data

    kwargs: dict[str, Any] = {}
    type_hints = {f.name: f.type for f in dataclasses.fields(cls)}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = type_hints.get(f.name)

        if value is None:
            kwargs[f.name] = None
            continue

        # 递归处理 list[X] 里 X 是 dataclass 的情况
        if isinstance(value, list) and isinstance(hint, str) and hint.startswith("list["):
            inner = hint[5:-1].strip()
            inner_cls = globals().get(inner)
            if dataclasses.is_dataclass(inner_cls):
                kwargs[f.name] = [_from_dict(inner_cls, v) for v in value]
                continue

        # 递归处理 Optional[X] / X 是 dataclass 的情况
        inner_name = hint.replace("Optional[", "").replace("]", "").strip() if isinstance(hint, str) else ""
        inner_cls = globals().get(inner_name)
        if dataclasses.is_dataclass(inner_cls) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(inner_cls, value)
            continue

        kwargs[f.name] = value

    return cls(**kwargs)


class DictMixin:
    """提供 to_dict / from_dict，方便 JSON 落盘与读回。"""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return _from_dict(cls, data)


# --------------------------------------------------------------------------
# 1. 文档级元数据：来源信息的"根"
# --------------------------------------------------------------------------
@dataclass
class DocumentMeta(DictMixin):
    """一份报告的全部身份信息。最终展示给用户的"公司/报告"就来自这里。"""

    doc_id: str                 # 形如 pingan_2025_annual，全局唯一
    company: str                # 中文简称，如 "中国平安"（展示用）
    company_full: str           # 中文全称（可选）
    company_key: str            # 配置里的英文 key，如 "pingan"
    stock_codes: list[str]      # ["601318.SH", "2318.HK"]
    exchange: str               # "上海证券交易所 / 香港交易所"
    report_year: int            # 2025
    report_name: str            # "2025年年度报告"（展示用）
    report_type: str            # annual | semi_annual
    source_file: str            # 原始 PDF 文件名
    source_path: str            # 原始 PDF 相对路径
    source_url: str = ""        # 真实下载地址，不要编造，没有就留空
    file_sha256: str = ""       # 原文件指纹，保证抽取结果可复现
    page_count: int = 0
    fetched_at: str = ""        # 下载时间 ISO8601


# --------------------------------------------------------------------------
# 2. 表格块：真正的行列结构 + 线性化文本
# --------------------------------------------------------------------------
@dataclass
class TableBlock(DictMixin):
    index: int                              # 本页内第几张表，从 1 开始
    page: int                               # 物理页序号（1-based）
    bbox: list[float] = field(default_factory=list)   # [x0, y0, x1, y1]
    n_rows: int = 0
    n_cols: int = 0
    # cells[r][c] -> str | None。None 表示该格在 PDF 里是空的/合并掉的。
    # 千万不要把 None 当成 0！
    cells: list[list[Optional[str]]] = field(default_factory=list)
    header: list[str] = field(default_factory=list)   # 表头行（推测）
    # 线性化文本，形如 "| 项目 | 2025年 |\n| --- | --- |\n| 营业收入 | 1,234 |"
    text: str = ""
    caption: str = ""                       # 表格上方/下方的标题文字（如"单位：人民币百万元"）
    extractor: str = ""                     # pymupdf | pdfplumber
    section: str = ""                       # 所属章节路径
    # "numeric"(以数据为主) | "text"(以文字为主) | "empty" | "unknown"
    # 只做标记，不据此删除任何表 —— 见 textnorm.table_profile 的说明
    quality: str = "unknown"
    numeric_ratio: float = 0.0              # 数字型单元格占比
    warnings: list[str] = field(default_factory=list)  # 如 "检测到疑似合并单元格"

    @property
    def empty(self) -> bool:
        return not self.text.strip()


# --------------------------------------------------------------------------
# 3. 文本块：段落 / 标题
# --------------------------------------------------------------------------
@dataclass
class TextBlock(DictMixin):
    index: int
    page: int
    bbox: list[float] = field(default_factory=list)
    text: str = ""
    kind: str = "paragraph"     # heading | paragraph | table_text | caption | other
    level: int = 0              # 标题层级：1=第X节, 2=一、, 3=（一）, 4=1.1.1；非标题为 0
    section: str = ""           # 该块所属的章节路径，如 "第三节 管理层讨论与分析 > 一、主要财务数据"
    font_size: float = 0.0
    font_name: str = ""
    bold: bool = False
    # 若该文本块落在某张表的区域内，这里指向表序号（1-based），
    # 用于后续去重；文本本身仍然保留，绝不丢弃。
    inside_table: int = 0
    printed_page: Optional[str] = None   # 页脚印刷页码（若识别到）


# --------------------------------------------------------------------------
# 4. 页级记录
# --------------------------------------------------------------------------
@dataclass
class PageRecord(DictMixin):
    page: int                   # 物理页序号，1-based
    width: float = 0.0
    height: float = 0.0
    printed_page: Optional[str] = None
    blocks: list[TextBlock] = field(default_factory=list)
    tables: list[TableBlock] = field(default_factory=list)
    # 该页的纯文本（按阅读顺序拼接），方便肉眼核对/调试
    text: str = ""
    char_count: int = 0
    # 页级告警，如字体编码缺失（抽出乱码）、扫描页需 OCR
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# 5. 整份文档的抽取结果：落盘为 data/extracted/<doc_id>.json
# --------------------------------------------------------------------------
@dataclass
class ExtractedDocument(DictMixin):
    meta: DocumentMeta
    pages: list[PageRecord] = field(default_factory=list)
    extractor: str = ""             # 主解析器名
    extractor_version: str = ""
    table_backend: str = ""
    extracted_at: str = ""
    body_font_size: float = 0.0     # 推断出的正文字号，标题识别基准
    stats: dict[str, Any] = field(default_factory=dict)

    def iter_blocks(self):
        for page in self.pages:
            for blk in page.blocks:
                yield page, blk

    def iter_tables(self):
        for page in self.pages:
            for tbl in page.tables:
                yield page, tbl


# --------------------------------------------------------------------------
# 6. 检索单元 Chunk：第二阶段的输入，也是最终展示来源的载体
# --------------------------------------------------------------------------
@dataclass
class Chunk(DictMixin):
    chunk_id: str               # 形如 pingan_2025_annual_p012_c003
    doc_id: str

    # ===== 作业要求的来源字段（回答里必须显示这五个）=====
    company: str                # 公司：中国平安
    report_year: int            # 2025
    report_name: str            # 报告：2025年年度报告
    page: int                   # 页码：12
    section: str                # 章节：第三节 管理层讨论与分析 > 一、主要财务数据
    text: str                   # 正文

    # ===== 附加溯源信息 =====
    page_end: int = 0           # 跨页 chunk 的结束页；不跨页时等于 page
    printed_page: Optional[str] = None
    block_type: str = "paragraph"   # paragraph | table | heading | mixed
    chunk_index: int = 0        # 同一 (doc, page, section) 内的顺序
    n_chars: int = 0
    stock_codes: list[str] = field(default_factory=list)
    source_file: str = ""
    tables: list[dict] = field(default_factory=list)  # 内嵌表格的 cells，便于精确取数
    # context_prefix 拼在 text 前面喂给 embedding，让向量带上上下文；
    # 展示给用户时只用 text，不要用 embed_text。
    context_prefix: str = ""

    @property
    def embed_text(self) -> str:
        return f"{self.context_prefix}\n{self.text}".strip() if self.context_prefix else self.text

    def citation(self, zh: bool = True) -> str:
        """生成作业要求的来源块。

        页码格式是作业指定的「页码：XX页」，不写成「第 XX 页」——
        引用块要能直接贴进报告，格式与要求逐字一致。
        """
        if zh:
            pages = (f"{self.page}页" if not self.page_end
                     or self.page_end == self.page
                     else f"{self.page}-{self.page_end}页")
            return (
                f"公司：{self.company}\n"
                f"报告：{self.report_name}\n"
                f"页码：{pages}\n"
                f"章节：{self.section or '（未识别）'}"
            )
        return (
            f"Company: {self.company}\n"
            f"Report: {self.report_name}\n"
            f"Page: {self.page}\n"
            f"Section: {self.section or 'N/A'}"
        )
