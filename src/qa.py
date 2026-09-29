"""问答编排：检索 → 判断材料够不够 → 交给大模型 → 带引用的答案（第三阶段）。

这一层**只调用 Phase 2 的 `Retriever.search()`**，不改动任何检索逻辑
（bm25 / hybrid / tokenize / retriever 一行都没动），也不碰索引。

闭环是：

    问题
      → Retriever.search()（混合召回，带公司过滤）
      → 判断材料够不够（下面第三段）
      → 把 chunk 正文 + 元数据拼成材料，交给大模型
      → 答案 + 引用（引用由程序从 chunk 元数据渲染，不由模型生成）

## 一、引用为什么一定可信

页面上的「公司 / 报告 / 页码」**全部来自 chunk 自己的元数据**
（`Chunk.company` / `report_name` / `page`），是第一阶段从 PDF 里抽出来时就记下的。
大模型只负责写答案正文，它没有机会改写引用块 —— 即使它在正文里编一个页码，
也不会变成引用。另外还会扫一遍正文，把材料里不存在的页码标成警告（见
`llm.find_suspicious_pages`），提示人工核对。

## 二、只截取、不改写

材料正文直接取自 chunk，**原样拼接，不做任何润色或数字格式化**。
`max_chars` 只做截尾（保留开头，因为关键数字常在前半段），
截断处会写明"已截断"，避免模型把半截话当成完整结论。

## 三、材料够不够：三层闸门

作业要求「检索不到可靠资料就明说，不要编造」。只靠一句提示词是不够的
（实测：无关问题"中国平安2026年的净利润是多少"的向量相似度高达 0.736，
比题库里最简单的一道题 0.629 还高，**单看相似度根本分不开**）。
所以这里是三层：

1. **相似度闸门**：top-1 余弦低于阈值就直接拒答，连模型都不调用（省 token）。
   阈值 0.58 是实测出来的，不是拍的 —— 见下面 MIN_COSINE 的注释。
2. **年份闸门**：问题里点名的年份，如果既不是任何材料的报告年份、
   也没在材料正文里出现过，就拒答并说明是年份对不上。
   这一层专门拦"2026年的净利润"这种：检索会返回 2025 年的数字，
   模型很可能顺手拿来答，那是实打实的答非所问。
3. **提示词闸门**：材料里没有的实体（比如"CEO 叫什么名字"），
   靠系统提示词要求模型明说"材料里没有"。

任何一层拦下来，都不会调用大模型 —— 与其让它硬编，不如直接说没有。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .llm import (LLMClient, LLMConfig, LLMError, Material, NO_MATERIAL_ANSWER,
                  SYSTEM_PROMPT, build_materials, build_user_prompt,
                  extract_citations, find_suspicious_pages)
from .retriever import MODE_HYBRID, Retriever

# top-1 余弦的拒答阈值。**实测标定，不要凭感觉改**：
#   题库（29 道应当能答的题）top-1 余弦：最低 0.629 / 中位 0.731
#   无关问题（红烧肉 0.296、小说 0.332、天气 0.404、量子纠缠 0.418、
#             世界杯 0.452、Python 0.447、美国GDP 0.529、比特币 0.555）
# 0.58 落在 0.555 和 0.629 中间，两边都留了余量。
# 注意"中国平安2026年的净利润"是 0.736 —— 它靠这一层拦不住，由年份闸门接手。
MIN_COSINE = float(os.environ.get("QA_MIN_COSINE") or 0.58)

YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")

# 拒答时的说明，措辞要和 NO_MATERIAL_ANSWER 一致：明确说"没答"而不是含糊带过
YEAR_MISMATCH_ANSWER = (
    "没有找到 {year} 年的数据，因此不回答。\n\n"
    "知识库收录的是 **2025 年**年报，检索到的材料里也没有出现 {year} 年 —— "
    "这种情况下拿 2025 年的数字来回答 {year} 年的问题就是答非所问，所以这里不作答。\n\n"
    "可以改问 2025 年，或换一家公司试试。"
)


@dataclass
class QAResult:
    """一次问答的完整结果。页面上要展示的东西都在这儿。"""

    question: str
    answer: str
    sources: list = field(default_factory=list)        # list[RetrievedChunk]
    materials: list[Material] = field(default_factory=list)
    answered_by_llm: bool = False
    refused: bool = False
    refusal_reason: str = ""                            # similarity / year / no_hits / error
    detected_companies: list[str] = field(default_factory=list)
    cited_indexes: list[int] = field(default_factory=list)
    suspicious_pages: list[int] = field(default_factory=list)
    top_cosine: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def has_answer(self) -> bool:
        return bool(self.answer) and not self.refused


def check_year(question: str, hits) -> int | None:
    """问题里的年份如果材料完全没有，返回那个年份；否则 None。

    为什么需要这一层：检索是"语义相似"，问 2026 年，它照样会返回 2025 年
    最像的段落 —— 相似度还很高。而模型看到"净利润"配上一堆真实数字，
    很容易直接拿来答。这属于**最危险的一类错误**：答案看着有出处、
    数字也是真的，但答的是另一年。
    """
    years = {int(y) for y in YEAR_RE.findall(question)}
    if not years:
        return None
    report_years = {getattr(h.chunk, "report_year", None) for h in hits}
    blob = "".join(h.chunk.text for h in hits)
    for y in sorted(years):
        if y in report_years:
            continue
        if str(y) in blob:          # 材料正文里提到过（比如同比对比 2024 年）
            continue
        return y
    return None


def _top_cosine(retriever: Retriever, question: str,
                allowed: list[bool] | None) -> float:
    """top-1 余弦，用作"材料够不够"的信号。"""
    if retriever.vector is None:
        return 1.0                  # 没装向量索引就不靠这一层判断
    got = retriever.vector.search(question, top_k=1, allowed=allowed)
    return float(got[0][1]) if got else 0.0


def answer_question(question: str, *,
                    retriever: Retriever | None = None,
                    top_k: int = 5,
                    mode: str = MODE_HYBRID,
                    company: str | None = None,
                    llm_config: LLMConfig | None = None,
                    llm_client: LLMClient | None = None,
                    min_cosine: float | None = None,
                    max_material_chars: int = 1200) -> QAResult:
    """完整问答流程。没有配置大模型时也能跑，只是不给答案正文。

    Args:
        top_k: 送给模型几条材料。5 条是折中：再多上下文变长，模型反而容易
            被无关段落带偏；再少可能漏掉关键那一页。
        llm_client: 测试时传假客户端，不联网。
    """
    r = retriever or Retriever.load()
    threshold = MIN_COSINE if min_cosine is None else min_cosine

    hits = r.search(question, top_k, company=company, mode=mode)
    detected = r.detect_companies(question)
    res = QAResult(question=question, answer="", sources=list(hits),
                   detected_companies=detected)

    # ---- 闸门 0：一条都没召回
    if not hits:
        res.refused, res.refusal_reason = True, "no_hits"
        res.answer = NO_MATERIAL_ANSWER
        return res

    # ---- 闸门 1：相似度太低（无关问题）
    allowed = r._mask_for(company) if company else (
        r._mask_for(detected[0]) if len(detected) == 1 else None)
    res.top_cosine = _top_cosine(r, question, allowed)
    if res.top_cosine < threshold:
        res.refused, res.refusal_reason = True, "similarity"
        res.answer = NO_MATERIAL_ANSWER
        res.notes.append(
            f"最高相似度 {res.top_cosine:.3f} 低于阈值 {threshold:.2f}，"
            f"判定为与年报无关的问题。")
        return res

    # ---- 闸门 2：年份对不上
    bad_year = check_year(question, hits)
    if bad_year is not None:
        res.refused, res.refusal_reason = True, "year"
        res.answer = YEAR_MISMATCH_ANSWER.format(year=bad_year)
        res.notes.append(f"问题问的是 {bad_year} 年，材料里没有这个年份。")
        return res

    # ---- 材料齐了，交给大模型
    res.materials = build_materials(hits, max_chars=max_material_chars)
    cfg = llm_config if llm_config is not None else LLMConfig.from_env()

    if not cfg.enabled:
        # 没配大模型不是错误：检索结果和引用照样能用，只是没有答案正文。
        # 这里**不会**用模板拼一段假答案 —— 那等于绕过"不编造"的要求。
        res.answer = ""
        res.refused = True
        res.refusal_reason = "no_llm"
        res.notes.append(
            "未配置大模型（缺 API key），只展示检索结果与引用，不生成答案。"
            "配置方法见 README 第三阶段一节。")
        return res

    client = llm_client or LLMClient(cfg)
    try:
        text = client.complete(SYSTEM_PROMPT,
                              build_user_prompt(question, res.materials))
    except LLMError as e:
        res.refused, res.refusal_reason = True, "error"
        res.answer = ""
        res.notes.append(f"调用大模型失败：{e}")
        return res

    res.answer = text
    res.answered_by_llm = True
    res.cited_indexes = extract_citations(text, len(res.materials))
    res.suspicious_pages = find_suspicious_pages(text, res.materials)
    if res.suspicious_pages:
        res.notes.append(
            "⚠️ 答案正文里出现了材料中没有的页码 "
            f"{res.suspicious_pages} —— 可能是模型自己写的，请以引用区为准。")
    if not res.cited_indexes:
        res.notes.append(
            "答案没有标注任何材料编号，无法确认它依据了哪几条材料，建议人工核对。")
    return res


__all__ = ["QAResult", "answer_question", "check_year", "MIN_COSINE",
           "YEAR_MISMATCH_ANSWER"]
