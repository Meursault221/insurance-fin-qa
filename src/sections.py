"""章节标题识别。

中文年报的章节层级通常长这样：

    第三节 管理层讨论与分析          <- level 1
    （一）主要财务数据               <- level 2 (括号中文数字)
    1. 营业收入                      <- level 3
    1.1 分险种保费收入               <- level 4

  港股的还常见：
    MANAGEMENT DISCUSSION AND ANALYSIS
    Results of Operations

本模块**只依赖文本 + 字号 + 是否加粗**，不依赖任何 PDF 库，
所以可以脱离 PDF 单独做单元测试。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CN_NUM = "一二三四五六七八九十百零〇两"

# --------------------------------------------------------------------------
# 标题正则。顺序即优先级，先匹配到的先用。
# 每个元素: (level, 正则, 说明)
# --------------------------------------------------------------------------
HEADING_RULES: list[tuple[int, re.Pattern, str]] = [
    # 第一级：第X节 / 第X章 / 第X部分
    (1, re.compile(rf"^第\s*[{CN_NUM}\d]+\s*[节章篇]"), "第X节/章"),
    # 第一级：重要提示、目录、释义、公司简介 等固定板块
    (1, re.compile(r"^(重要提示|目录|释义|公司简介|董事[会事]报告|"
                   r"管理层讨论与分析|财务报告|审计报告|公司治理|"
                   r"环境[、,]?社会及管治报告|企业管治报告)\s*$"), "固定板块名"),
    # 第二级：（一）xxx
    (2, re.compile(rf"^[（(]\s*[{CN_NUM}\d]+\s*[）)]"), "（一）"),
    # 第二级：一、xxx
    (2, re.compile(rf"^[{CN_NUM}]+\s*[、.．]\s*\S"), "一、"),
    # 第四级：1.1.1（要先于 1.1 判断）
    (4, re.compile(r"^\d+\s*[.．]\s*\d+\s*[.．]\s*\d+"), "1.1.1"),
    # 第三级：1.1
    (3, re.compile(r"^\d+\s*[.．]\s*\d+(?!\d)"), "1.1"),
    # 第三级：1、xxx  /  1. xxx
    (3, re.compile(r"^\d+\s*[、.．]\s*\S"), "1、"),
    # 附注类
    (3, re.compile(rf"^附注\s*[{CN_NUM}\d]+"), "附注X"),
    # 英文标题（港股）
    (2, re.compile(r"^[A-Z][A-Za-z\s,&\-]{3,60}$"), "English heading"),
]

# 明显是正文而不是标题的特征：以句号/分号结尾、太长、数字太多
_SENTENCE_END = re.compile(r"[。；;：:]\s*$")
_NUM_TOKEN = re.compile(r"[\d,]+(?:\.\d+)?")

# 整段被括号包住的短文本（"（除特别注明外，金额单位为人民币百万元）"、"（左坐标）"）。
# 这类是注解/单位说明，不是标题 —— 它们当标题会污染章节路径。
# 注意 "（一）主要财务数据" 不以 ） 结尾，不会被这条误伤。
_FULLY_PARENTHESIZED = re.compile(r"^[（(][^（(）)]*[）)]$")

# 以虚词/连接词/顿号结尾的绝不是标题 —— 这是被排版折行截断的正文片段。
# 实测平安年报里 "……已经建立了依托本土优势兼具国际标准的" 被当成了
# 一级标题，并在标题栈里滞留了 52 个 chunk。
_SENTENCE_TAIL = re.compile(r"[的了和与及或地得等并且、，,]\s*$")

# 又长又带逗号的，是句子而不是标题（标题不会写这么啰嗦）
_CLAUSE_COMMA = re.compile(r"[，,]")
LONG_CANDIDATE_CHARS = 20

# "105.1亿元"、"67.79%收购" 这类是数量值，不是章节号 ——
# 它们长得像 "1.1" 式编号，会骗过编号正则。判据是数字后面紧跟计量单位。
_IS_QUANTITY = re.compile(
    r"^[\d,，.]+\s*(?:亿元|万元|千元|百万元|元|%|％|倍|个百分点|万人|人|个|天|年|月|日)")


MAX_HEADING_CHARS = 40
MIN_HEADING_CHARS = 2


def _numeric_density(text: str) -> float:
    """数字字符占比。用来把 "1,234.56 1,111.00" 这种数据行排除掉。"""
    if not text:
        return 0.0
    digits = sum(1 for ch in text if ch.isdigit() or ch in ",.%()-")
    return digits / len(text)


# 编号前缀（"第三节" / "（一）" / "1.1.1" / "1、" …）。
# 判断数字密度前要先把它剥掉 —— 否则 "1.1.1 详细说明" 会因为
# 编号本身占了太多字符而被误判成数据行。
_NUMBER_PREFIX = re.compile(
    rf"^(?:第\s*[{CN_NUM}\d]+\s*[节章篇]"
    rf"|附注\s*[{CN_NUM}\d]+"
    rf"|[（(]\s*[{CN_NUM}\d]+\s*[）)]"
    rf"|[{CN_NUM}]+\s*[、.．]"
    rf"|\d+\s*[.．]\s*\d+(?:\s*[.．]\s*\d+)*"
    rf"|\d+\s*[、.．])\s*"
)

# 标题的正文部分至少要有这些字符之一，否则说明剥掉编号后就不剩东西了
_HAS_WORD = re.compile(r"[一-鿿A-Za-z]")


def split_number_prefix(text: str) -> tuple[str, str]:
    """返回 (编号前缀, 剩余正文)。没有编号时前缀为空串。"""
    m = _NUMBER_PREFIX.match(text)
    if not m:
        return "", text
    return m.group(0).strip(), text[m.end():].strip()


def classify_block(
    text: str,
    font_size: float = 0.0,
    body_font_size: float = 0.0,
    bold: bool = False,
    allow_font_size_rule: bool = True,
) -> tuple[str, int]:
    """返回 (kind, level)。kind ∈ {"heading", "paragraph"}，level 为 0 表示非标题。"""
    t = (text or "").strip()
    if not t:
        return "other", 0

    if len(t) > MAX_HEADING_CHARS or len(t) < MIN_HEADING_CHARS:
        return "paragraph", 0
    if _SENTENCE_END.search(t):
        return "paragraph", 0
    if _FULLY_PARENTHESIZED.match(t):
        return "paragraph", 0
    if _SENTENCE_TAIL.search(t):
        return "paragraph", 0
    if len(t) >= LONG_CANDIDATE_CHARS and _CLAUSE_COMMA.search(t):
        return "paragraph", 0
    if _IS_QUANTITY.match(t):
        return "paragraph", 0

    _prefix, body = split_number_prefix(t)
    # 剥掉编号后必须还剩实际的文字 —— 只有编号的 "1.23"、"(1,234)" 不是标题
    if not _HAS_WORD.search(body):
        return "paragraph", 0
    # 数字密度只看正文部分，否则 "1.1.1 详细说明" 会被编号拖高而被误杀。
    # 这一步专门过滤 "1,234.56 1,111.00 11.1" 这类数据行。
    density_text = body if _prefix else t
    if _numeric_density(density_text) > 0.35:
        return "paragraph", 0

    # 1) 正则规则
    for level, pattern, _name in HEADING_RULES:
        if pattern.match(t):
            # "1、xxx" 这种如果同时数字很多，已经在上面的密度判断里被拦掉了
            return "heading", level

    # 2) 字号 / 加粗规则（作为兜底）
    if allow_font_size_rule and body_font_size > 0 and font_size > 0:
        ratio = font_size / body_font_size
        if ratio >= 1.30:
            return "heading", 1
        if ratio >= 1.12 or (bold and ratio >= 1.04):
            return "heading", 2

    return "paragraph", 0


# --------------------------------------------------------------------------
# 章节路径追踪：把一串 block 的层级整理成 "A > B > C"
# --------------------------------------------------------------------------
@dataclass
class SectionTracker:
    """维护一个标题栈，为每个 block 计算它所属的章节路径。

    用法：
        tr = SectionTracker()
        tr.update("heading", 1, "第三节 管理层讨论与分析")  -> "第三节 管理层讨论与分析"
        tr.update("heading", 2, "（一）主要财务数据")        -> "第三节 ... > （一）主要财务数据"
        tr.update("paragraph", 0, "本集团 2025 年 ...")      -> 同上（正文沿用栈顶）
    """

    separator: str = " > "
    max_depth: int = 3
    stack: list[tuple[int, str]] = field(default_factory=list)

    def update(self, kind: str, level: int, text: str) -> str:
        if kind == "heading" and level > 0:
            t = text.strip()
            # 弹出所有层级 >= 当前层级的标题
            while self.stack and self.stack[-1][0] >= level:
                self.stack.pop()
            self.stack.append((level, t))
            if len(self.stack) > self.max_depth:
                self.stack = self.stack[-self.max_depth:]
        return self.current

    @property
    def current(self) -> str:
        # 去重相邻重复（PDF 里标题偶尔会重复出现，比如跨页续排）
        parts: list[str] = []
        for _lv, t in self.stack:
            if not parts or parts[-1] != t:
                parts.append(t)
        return self.separator.join(parts)

    def reset(self) -> None:
        self.stack.clear()


# 常见章节别名，第二阶段做"按章节过滤"时可以用
KNOWN_SECTIONS = [
    "重要提示", "释义", "公司简介", "经营情况讨论与分析", "管理层讨论与分析",
    "公司治理", "董事、监事、高级管理人员", "财务报告", "审计报告",
    "合并资产负债表", "合并利润表", "合并现金流量表", "主要财务数据",
    "保费收入", "偿付能力", "风险管理", "内含价值",
]
