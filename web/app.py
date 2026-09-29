"""第三阶段：财报问答页面（Streamlit）。

启动：
    streamlit run web/app.py

页面做的事：输入问题 → 混合检索 → 把命中的 chunk 交给大模型 → 显示答案 + 引用。
**引用（公司 / 报告 / 页码）直接取自 chunk 元数据，不由大模型生成。**

检索逻辑一律走 Phase 2 的 `src/retriever.py`，这个文件里没有任何检索实现。
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm import LLMConfig, load_env_file          # noqa: E402
from src.qa import MIN_COSINE, answer_question        # noqa: E402
from src.retriever import MODE_ZH, Retriever          # noqa: E402

st.set_page_config(page_title="上市保险公司财报问答", page_icon="📊",
                   layout="wide")

EXAMPLES = [
    "中国平安2025年归属于母公司股东的净利润是多少？",
    "中国太保2025年末的总资产规模是多少？",
    "众安在线2025年的承保综合成本率是多少？",
    "比较中国平安和中国人寿2025年的归母净利润",
    "今天天气怎么样？",                      # 用来演示"检索不到就不编"
]


@st.cache_resource(show_spinner="正在加载索引与向量模型（首次约 20 秒）…")
def get_retriever() -> Retriever:
    """整个进程只加载一次。Streamlit 每次交互都会重跑脚本，
    不缓存的话每问一句都要重新加载 95MB 的向量模型。"""
    return Retriever.load()


def main() -> None:
    st.title("📊 上市保险公司财报问答")
    st.caption(
        "10 家上市保险公司 · 2025 年年报 · 混合检索（BM25 + 向量）"
        "· 每个答案都能追溯到公司、报告和页码"
    )

    load_env_file(ROOT / ".env")
    cfg = LLMConfig.from_env()

    # ---------------------------------------------------------------- 侧栏
    with st.sidebar:
        st.header("设置")
        mode = st.radio("检索方式", ["hybrid", "bm25", "vector"],
                        format_func=lambda m: MODE_ZH[m], index=0,
                        help="hybrid = BM25 + 向量，用 RRF 融合")
        top_k = st.slider("送给模型的片段数", 3, 10, 5,
                          help="太少可能漏掉关键那页，太多容易被无关段落带偏")
        company = st.text_input("限定公司（可留空）",
                               placeholder="如 pingan / cpic / picc_pc",
                               help="留空则从问题里自动识别公司")
        st.divider()
        st.subheader("大模型")
        if cfg.enabled:
            st.success(f"已配置：{cfg.provider} / {cfg.model}")
        else:
            st.warning("未配置 —— 只显示检索结果与引用，不生成答案")
            st.caption(
                "在项目根目录建一个 `.env` 文件，写一行：\n\n"
                "`ANTHROPIC_API_KEY=sk-ant-...`\n\n"
                "或任意 OpenAI 兼容接口：\n\n"
                "`OPENAI_API_KEY=sk-...`\n\n"
                "`LLM_BASE_URL=https://api.deepseek.com/v1`")
        st.divider()
        st.caption(f"相似度拒答阈值：{MIN_COSINE:.2f}")
        show_chunks = st.checkbox("显示检索到的原文片段", value=True)

    # ---------------------------------------------------------------- 提问
    def _use_example(text: str) -> None:
        # 用回调而不是 `if button:` —— 回调在脚本重跑**之前**执行，
        # 这时改 st.session_state 不会撞上"控件已实例化"那个报错。
        st.session_state.question = text

    question = st.text_input(
        "请输入关于这 10 家公司 2025 年年报的问题",
        key="question", placeholder=EXAMPLES[0])

    cols = st.columns(len(EXAMPLES))
    for c, ex in zip(cols, EXAMPLES):
        c.button(ex[:14] + "…", help=ex, use_container_width=True,
                 on_click=_use_example, args=(ex,))

    if not question.strip():
        st.info("输入一个问题，或点上面的示例。"
                "示例题的答案在年报「财务摘要」一节（第 14–15 页也列了归母净利润）。")
        return

    retriever = get_retriever()

    with st.spinner("检索并生成回答…"):
        result = answer_question(question, retriever=retriever, top_k=top_k,
                                 mode=mode, company=(company.strip() or None))

    st.divider()

    # ---------------------------------------------------------------- 答案
    st.subheader("答案")
    if result.answered_by_llm:
        st.markdown(result.answer)
    elif result.refusal_reason == "no_llm":
        st.warning("检索已完成，但没有配置大模型，因此不生成答案。"
                   "下方是命中的年报片段与引用，可自行查阅。")
    elif result.refusal_reason == "error":
        # 这一档是**真的出错了**（网络/接口/key），和拒答不是一回事
        st.error("调用大模型失败，没有生成答案。")
    elif result.refused:
        # 拒答是**正确行为**，不是故障 —— 用 info 而不是 error 展示
        st.info(result.answer)
    else:
        st.write(result.answer or "（没有生成答案）")

    for note in result.notes:
        st.caption(note)

    # ---------------------------------------------------------------- 引用
    # 没有答案时，这些片段**没有**被用作依据，标题必须说清楚 ——
    # 否则"拒答"旁边摆 5 条来源，看起来像在暗示它们支撑了某个结论。
    if result.answered_by_llm:
        st.subheader(f"引用来源（{len(result.sources)} 条）")
    else:
        st.subheader(f"检索到的片段（{len(result.sources)} 条，未采用）")
        st.caption(
            "本次没有生成答案，下面这些只是检索结果，**没有被当作答案依据**，仅供参考。"
            if result.refusal_reason != "no_llm" else
            "没有配置大模型，下面这些只是检索结果，**没有被用于生成答案**。")
    if result.detected_companies:
        st.caption("识别到的公司：" + "、".join(result.detected_companies))
    if not result.sources:
        st.write("（没有检索到相关内容）")
        return

    for i, hit in enumerate(result.sources, start=1):
        ch = hit.chunk
        mark = "✅ " if i in result.cited_indexes else ""
        with st.expander(f"{mark}[{i}] {ch.company}｜{ch.report_name}"
                         f"｜第 {ch.page} 页", expanded=(i == 1)):
            # 引用块直接用 Chunk.citation()，和作业要求的格式一致
            st.code(ch.citation(), language=None)
            meta = st.columns(4)
            meta[0].metric("融合得分", f"{hit.score:.4f}")
            meta[1].metric("检索方式", MODE_ZH.get(hit.source, hit.source))
            meta[2].metric("类型", ch.block_type or "-")
            meta[3].metric("字数", ch.n_chars or len(ch.text))
            if ch.stock_codes:
                st.caption("股票代码：" + "、".join(ch.stock_codes))
            st.caption(f"chunk_id：`{ch.chunk_id}`")
            if show_chunks:
                st.text_area("原文片段", ch.text, height=180,
                             key=f"txt_{i}_{ch.chunk_id}", disabled=True)


if __name__ == "__main__":
    main()
