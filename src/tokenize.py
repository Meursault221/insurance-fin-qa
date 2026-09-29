"""中文分词 —— BM25 的地基。

年报文本有三个特点，通用分词器直接拿来用会出问题：

1. **数字是主要内容**。"2,027,592" 被切开就再也匹配不回原数，
   而本项目的问答大量是数字问题。所以数字串先单独识别成整 token，
   并且**去掉千分位逗号** —— 这样用户写 "2027592" 也能命中报告里的 "2,027,592"。

2. **专有名词多**。公司简称、产品名（"人保健康""众安在线"）未必进得了词典。
   补一层字符 bigram 兜底：只要两个字相邻出现过，就能被召回。

3. **简繁混杂**。10 家里有 5 家是港股年报（中国人保、中国太平、中国再保险、
   中国财险、众安在线），正文是繁体，合计 3937/7507 条 chunk，占 52%。
   用户用简体问"总资产"，字面上永远匹配不上原文的"總資產" —— 实测这道题
   BM25 三路全挂，而它本来是最简单的关键词题。所以建索引和检索前
   统一转成简体。

**转换只发生在 token 层，chunk 原文一个字都不动。** 引用里的公司名、页码、
数字仍然是年报原文（繁体就繁体），因为引用直接取自 data/chunks。
"""

from __future__ import annotations

import re

try:
    import zhconv
except ImportError:                      # pragma: no cover
    zhconv = None

# 数字串：可带千分位、小数点、百分号、负号
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
# 连续的中文（含常见异体字区间）
_CJK = re.compile(r"[㐀-䶿一-鿿]+")
# 连续的英文/数字混合词
_LATIN = re.compile(r"[a-zA-Z][a-zA-Z0-9\-_]*")

# 中文停用词。年报里"的""和"这类词对区分度没有贡献，
# 但它们在 BM25 里会被 IDF 自动压低 —— 除非某词几乎每篇都有。
# 这里只列真正无信息量的功能词。
STOPWORDS = {
    "的", "了", "和", "与", "及", "或", "在", "是", "为", "对", "以", "等",
    "本", "该", "各", "有", "不", "也", "而", "中", "上", "下", "之", "其",
    "我们", "公司", "集团",  # "公司""集团"几乎每 chunk 都有，无区分度
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "is", "are",
}


def _normalize_number(tok: str) -> str:
    """去掉千分位逗号，保留小数点和百分号。"""
    return tok.replace(",", "").lower()


def to_simplified(text: str) -> str:
    """繁体 → 简体。

    缺 zhconv 时**报错而不是跳过** —— 静默跳过会让 52% 的语料（5 家港股年报）
    悄悄检索不到，表现为"某些公司怎么问都问不出来"，这种失败很难归因。
    """
    if zhconv is None:                   # pragma: no cover
        raise ImportError(
            "缺少 zhconv，无法做简繁归一。10 家公司里有 5 家是港股年报，"
            "正文为繁体，跳过这一步会让它们检索不到。\n"
            "  安装：python -m pip install zhconv")
    return zhconv.convert(text, "zh-cn")


def tokenize(text: str, *, bigrams: bool = True) -> list[str]:
    """把一段文本切成 token 列表。

    Args:
        text: 原文
        bigrams: 是否为中文补字符 bigram（提升专有名词召回）
    """
    if not text:
        return []

    text = to_simplified(text)
    tokens: list[str] = []
    # 按"数字 / 中文 / 英文"三类交替扫描，保证数字不被中文分词器切碎。
    # 注意用 .pattern 取出模式串 —— 直接把 Pattern 对象插进 f-string 会得到
    # "re.compile('...')" 的字面量，正则什么也匹配不到，分词结果恒为空。
    for m in re.finditer(rf"{_NUM.pattern}|{_CJK.pattern}|{_LATIN.pattern}", text):
        piece = m.group(0)
        if _NUM.fullmatch(piece):
            tokens.append(_normalize_number(piece))
        elif _CJK.fullmatch(piece):
            if len(piece) == 1:
                tokens.append(piece)
                continue
            import jieba
            for w in jieba.lcut(piece):
                w = w.strip()
                if not w:
                    continue
                tokens.append(w)
                if bigrams and len(w) > 2:
                    # 词内 bigram：让"众安在线"也能被"众安"召回。
                    # 只对 len>2 的词补 —— 两字词的 bigram 就是它自己，
                    # 补进去只是把 tf 翻倍，白白抬高了这些词的权重。
                    tokens.extend(w[i:i + 2] for i in range(len(w) - 1))
        else:
            tokens.append(piece.lower())

    return [t for t in tokens if t and t not in STOPWORDS]


def tokenize_query(text: str, *, bigrams: bool = True) -> list[str]:
    """检索词的分词。与文档侧用同一套规则，否则 BM25 统计对不上。"""
    return tokenize(text, bigrams=bigrams)


def explain(text: str) -> str:
    """调试用：把分词结果摊开看，便于核对为什么某个 chunk 没被召回。"""
    return " | ".join(tokenize(text))
