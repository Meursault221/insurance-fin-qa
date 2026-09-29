"""第三阶段问答层的测试：材料拼装、引用核查、三层闸门、防伪造。

**全部离线** —— 用假检索器和假 transport，不读 data/chunks、不连网、不调模型。
真实数据上的效果由 scripts/eval_retrieval.py（检索层）和人工跑页面（问答层）衡量。

这里最要紧的一条是 `TestCitationsComeFromMetadata`：引用必须来自 chunk 元数据，
模型在正文里写什么都不影响引用块。这是整个第三阶段"不编造来源"的落点。
"""

from __future__ import annotations

import pytest

from src.chunker import Chunk
from src.llm import (LLMClient, LLMConfig, LLMError, Material, build_materials,
                     build_user_prompt, extract_citations,
                     find_suspicious_pages, load_env_file, _scrub)
from src.qa import QAResult, answer_question, check_year
from src.retriever import RetrievedChunk


# ---------------------------------------------------------------- 测试替身
def make_chunk(cid: str = "pingan_2025_annual_p0040_c0081", *,
               company: str = "中国平安", page: int = 40,
               text: str = "归属于母公司股东的净利润 134,778 百万元",
               section: str = "财务摘要", report_name: str = "2025年年度报告",
               report_year: int = 2025, page_end: int = 0,
               block_type: str = "mixed") -> Chunk:
    return Chunk(chunk_id=cid, doc_id="pingan_2025_annual", company=company,
                 report_year=report_year, report_name=report_name, page=page,
                 section=section, text=text, page_end=page_end,
                 block_type=block_type)


def make_hit(chunk: Chunk, *, score: float = 1.0, rank: int = 1,
             source: str = "hybrid", company_key: str = "pingan") -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, score=score, rank=rank, source=source,
                          company_key=company_key)


class FakeVector:
    """只回答 top-1 余弦，`_top_cosine` 就靠它。"""

    def __init__(self, cosine: float):
        self.cosine = cosine
        self.calls: list = []

    def search(self, query, top_k=1, allowed=None):
        self.calls.append((query, top_k, allowed))
        return [(0, self.cosine)]


class FakeRetriever:
    """假检索器。记下调用参数，方便断言"检索层被正确调用"。"""

    def __init__(self, hits=None, *, cosine: float = 0.9,
                 detected: list[str] | None = None):
        self.hits = list(hits or [])
        self.vector = FakeVector(cosine)
        self._detected = detected or []
        self.search_calls: list[dict] = []

    def search(self, query, top_k=5, **kw):
        self.search_calls.append({"query": query, "top_k": top_k, **kw})
        return self.hits[:top_k]

    def detect_companies(self, question):
        return list(self._detected)

    def _mask_for(self, company):
        return [company == "pingan"] * 3


def fake_client(text: str = "归母净利润 134,778 百万元 [1]", *,
                error: Exception | None = None):
    """假大模型客户端，同时记录收到的提示词。"""

    class _C:
        def __init__(self):
            self.calls: list[tuple[str, str]] = []

        def complete(self, system, user):
            self.calls.append((system, user))
            if error is not None:
                raise error
            return text

    return _C()


LLM_ON = LLMConfig(provider="anthropic", model="claude-sonnet-5",
                   api_key="sk-ant-test", base_url="https://example.invalid")


# ---------------------------------------------------------------- 配置
class TestLLMConfig:
    def test_没有key就是关闭(self):
        cfg = LLMConfig.from_env({})
        assert cfg.provider == "none" and not cfg.enabled

    def test_有anthropic_key就自动选anthropic(self):
        cfg = LLMConfig.from_env({"ANTHROPIC_API_KEY": "sk-ant-x"})
        assert cfg.provider == "anthropic" and cfg.enabled
        assert cfg.model == "claude-sonnet-5"

    def test_有openai_key就自动选openai(self):
        cfg = LLMConfig.from_env({"OPENAI_API_KEY": "sk-x"})
        assert cfg.provider == "openai" and cfg.enabled
        assert cfg.base_url == "https://api.deepseek.com/v1"
        assert cfg.model == "deepseek-chat"

    def test_显式provider优先于自动判断(self):
        cfg = LLMConfig.from_env({"LLM_PROVIDER": "openai",
                                  "OPENAI_API_KEY": "sk-x",
                                  "ANTHROPIC_API_KEY": "sk-ant-y"})
        assert cfg.provider == "openai" and cfg.api_key == "sk-x"

    def test_自定义base_url去掉末尾斜杠(self):
        cfg = LLMConfig.from_env({"LLM_PROVIDER": "openai",
                                  "OPENAI_API_KEY": "sk-x",
                                  "LLM_BASE_URL": "http://127.0.0.1:11434/v1/"})
        assert cfg.base_url == "http://127.0.0.1:11434/v1"

    def test_显式关闭(self):
        cfg = LLMConfig.from_env({"LLM_PROVIDER": "none",
                                  "ANTHROPIC_API_KEY": "sk-ant-x"})
        assert not cfg.enabled

    def test_不认识的provider要报错(self):
        with pytest.raises(LLMError):
            LLMConfig.from_env({"LLM_PROVIDER": "gemini",
                                "ANTHROPIC_API_KEY": "sk-ant-x"})

    def test_有provider但没key不算enabled(self):
        cfg = LLMConfig.from_env({"LLM_PROVIDER": "anthropic"})
        assert cfg.provider == "anthropic" and not cfg.enabled


# ---------------------------------------------------------------- 材料拼装
class TestBuildMaterials:
    def test_编号从1开始且带全部来源字段(self):
        hits = [make_hit(make_chunk(page=40), rank=1),
                make_hit(make_chunk(cid="c2", page=15), rank=2)]
        ms = build_materials(hits)
        assert [m.index for m in ms] == [1, 2]
        assert ms[0].company == "中国平安"
        assert ms[0].report_name == "2025年年度报告"
        assert ms[0].page == 40

    def test_header格式包含作业要求的四项(self):
        m = build_materials([make_hit(make_chunk())])[0]
        h = m.header()
        assert h.startswith("[1] ")
        for key in ("公司：", "报告：", "页码：", "章节："):
            assert key in h

    def test_引用块与Chunk_citation格式一致(self):
        ch = make_chunk(page=40, page_end=41)
        m = build_materials([make_hit(ch)])[0]
        assert m.citation() == ch.citation()

    def test_跨页chunk页码显示成区间(self):
        m = build_materials([make_hit(make_chunk(page=40, page_end=41))])[0]
        assert "40-41页" in m.header()

    def test_超长只截尾不截头(self):
        head = "开头这几个字必须留下"
        ch = make_chunk(text=head + "中" * 5000)
        m = build_materials([make_hit(ch)], max_chars=100)[0]
        assert m.text.startswith(head)
        assert len(m.text) < 200
        assert "已截断" in m.text

    def test_不超长就不加截断标记(self):
        m = build_materials([make_hit(make_chunk(text="很短"))])[0]
        assert "已截断" not in m.text

    def test_正文原样拼接不改数字(self):
        ch = make_chunk(text="总资产 1,234,567.89 元")
        m = build_materials([make_hit(ch)])[0]
        assert "1,234,567.89" in m.text      # 千分位和单位都不能动

    def test_空白被压成单空格(self):
        m = build_materials([make_hit(make_chunk(text="甲\n\n  乙\t丙"))])[0]
        assert m.text == "甲 乙 丙"


class TestBuildUserPrompt:
    def test_包含问题和所有材料的头部(self):
        ms = [Material(index=1, company="中国平安", report_name="2025年年度报告",
                       page=40, page_end=0, section="财务摘要", text="甲"),
              Material(index=2, company="中国人寿", report_name="2025年年度报告",
                       page=7, page_end=0, section="", text="乙")]
        p = build_user_prompt("谁最赚钱？", ms)
        assert "谁最赚钱？" in p
        assert "[1] 公司：中国平安" in p and "[2] 公司：中国人寿" in p
        assert "共 2 条" in p
        assert "（未识别）" in p          # 章节为空时的占位

    def test_没有材料也不崩(self):
        assert "没有任何材料" in build_user_prompt("测试", [])

    def test_材料正文原样进提示词(self):
        ms = [Material(index=1, company="X", report_name="R", page=1,
                       page_end=0, section="S", text="保费收入 1,803.68 亿元")]
        assert "1,803.68 亿元" in build_user_prompt("q", ms)


# ---------------------------------------------------------------- 引用核查
class TestCitations:
    def test_去重且保序(self):
        assert extract_citations("甲 [1] 乙 [3] 丙 [1] 丁 [2]", 5) == [1, 3, 2]

    def test_越界的编号丢掉(self):
        assert extract_citations("甲 [1] 乙 [9]", 3) == [1]

    def test_没有引用返回空(self):
        assert extract_citations("没有任何编号", 3) == []

    def test_空回答不崩(self):
        assert extract_citations("", 3) == [] and extract_citations(None, 3) == []

    def test_三位的编号不算引用(self):
        # [123] 更像年份/编号，材料数不会到三位
        assert extract_citations("[123]", 200) == []


class TestSuspiciousPages:
    def test_材料里没有的页码被标出(self):
        ms = build_materials([make_hit(make_chunk(page=40))])
        assert find_suspicious_pages("见第 128 页", ms) == [128]

    def test_材料里有的页码不算可疑(self):
        ms = build_materials([make_hit(make_chunk(page=40))])
        assert find_suspicious_pages("见第 40 页", ms) == []

    def test_跨页区间内的页码不算可疑(self):
        ms = build_materials([make_hit(make_chunk(page=40, page_end=42))])
        assert find_suspicious_pages("见第 41 页", ms) == []

    def test_金额不会被误判成页码(self):
        ms = build_materials([make_hit(make_chunk(page=40))])
        assert find_suspicious_pages("净利润 134,778 百万元", ms) == []

    def test_多个可疑页码去重(self):
        ms = build_materials([make_hit(make_chunk(page=40))])
        assert find_suspicious_pages("见 128 页，另见第 99 页和第 128 页", ms) == [128, 99]


class TestScrub:
    def test_key被抹掉(self):
        assert "sk-ant-abcdef123" not in _scrub("失败 sk-ant-abcdef123456")

    def test_普通文本不受影响(self):
        assert _scrub("净利润 134,778") == "净利润 134,778"


# ---------------------------------------------------------------- HTTP 层
class TestLLMClient:
    def test_anthropic解析出文本块(self):
        seen = {}

        def transport(url, payload, headers, timeout):
            seen.update(url=url, payload=payload, headers=headers)
            return {"content": [{"type": "text", "text": "答案 [1]"}]}

        c = LLMClient(LLM_ON, transport=transport)
        assert c.complete("sys", "user") == "答案 [1]"
        assert seen["url"] == "https://example.invalid/v1/messages"
        assert seen["headers"]["x-api-key"] == "sk-ant-test"
        assert seen["payload"]["system"] == "sys"

    def test_openai解析出choices(self):
        seen = {}
        cfg = LLMConfig(provider="openai", model="deepseek-chat",
                        api_key="sk-x", base_url="https://api.deepseek.com/v1")

        def transport(url, payload, headers, timeout):
            seen.update(url=url, headers=headers)
            return {"choices": [{"message": {"content": "答案"}}]}

        assert LLMClient(cfg, transport=transport).complete("s", "u") == "答案"
        assert seen["url"] == "https://api.deepseek.com/v1/chat/completions"
        assert seen["headers"]["Authorization"] == "Bearer sk-x"

    def test_没配key就报错(self):
        with pytest.raises(LLMError):
            LLMClient(LLMConfig(provider="none")).complete("s", "u")

    def test_接口没返回文本要报错(self):
        c = LLMClient(LLM_ON, transport=lambda *a, **k: {"content": []})
        with pytest.raises(LLMError):
            c.complete("s", "u")

    def test_报错信息里不能出现key(self):
        def boom(*a, **k):
            raise LLMError("失败 sk-ant-abcdef1234567890")

        with pytest.raises(LLMError) as e:
            LLMClient(LLM_ON, transport=boom).complete("s", "u")
        assert "sk-ant-abcdef1234567890" not in str(e.value)


class TestLoadEnvFile:
    def test_读文件填进环境变量(self, tmp_path, monkeypatch):
        monkeypatch.delenv("INS_FIN_QA_TEST", raising=False)
        f = tmp_path / ".env"
        f.write_text("# 注释\nINS_FIN_QA_TEST=hello\n", encoding="utf-8")
        assert load_env_file(f) == ["INS_FIN_QA_TEST"]

    def test_已存在的环境变量不被覆盖(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INS_FIN_QA_TEST", "命令行设的")
        f = tmp_path / ".env"
        f.write_text("INS_FIN_QA_TEST=文件里的\n", encoding="utf-8")
        assert load_env_file(f) == []

    def test_文件不存在返回空(self, tmp_path):
        assert load_env_file(tmp_path / "nope.env") == []

    def test_引号被去掉(self, tmp_path, monkeypatch):
        import os
        monkeypatch.delenv("INS_FIN_QA_TEST2", raising=False)
        f = tmp_path / ".env"
        f.write_text('INS_FIN_QA_TEST2="带引号的值"\n', encoding="utf-8")
        load_env_file(f)
        assert os.environ["INS_FIN_QA_TEST2"] == "带引号的值"

    def test_返回的只有变量名不含值(self, tmp_path, monkeypatch):
        monkeypatch.delenv("INS_FIN_QA_TEST3", raising=False)
        f = tmp_path / ".env"
        f.write_text("INS_FIN_QA_TEST3=sekrit\n", encoding="utf-8")
        assert load_env_file(f) == ["INS_FIN_QA_TEST3"]   # 不含 "sekrit"

    def test_默认路径锚在项目根而不是当前目录(self):
        """不传路径时要读项目根的 .env，不是 CWD 的。

        锚错地方的后果很隐蔽：从别的目录跑会读不到（提示"未配置大模型"），
        而如果那个目录恰好也有个 .env，就会读到别人的配置。
        """
        from src.llm import DEFAULT_ENV_PATH
        assert DEFAULT_ENV_PATH.name == ".env"
        assert (DEFAULT_ENV_PATH.parent / "src" / "llm.py").exists()


# ---------------------------------------------------------------- 年份闸门
class TestCheckYear:
    def test_年份出现在报告年份里就放行(self):
        hits = [make_hit(make_chunk(report_year=2025, text="2025年数据"))]
        assert check_year("中国平安2025年的净利润？", hits) is None

    def test_年份在正文里提到过也放行(self):
        # 比如"2024年"作为同比对比出现在正文里
        hits = [make_hit(make_chunk(report_year=2025, text="较2024年增长 5%"))]
        assert check_year("2024年的净利润？", hits) is None

    def test_年份完全查不到就拦下(self):
        hits = [make_hit(make_chunk(report_year=2025, text="本期净利润 100"))]
        assert check_year("中国平安2026年的净利润？", hits) == 2026

    def test_问题里没有年份就不拦(self):
        hits = [make_hit(make_chunk(report_year=2025, text="净利润 100"))]
        assert check_year("中国平安的净利润？", hits) is None

    def test_不会把长数字里的四位当成年份(self):
        hits = [make_hit(make_chunk(report_year=2025, text="净利润 100"))]
        assert check_year("净利润 12345678 元？", hits) is None


# ---------------------------------------------------------------- 编排
class TestAnswerQuestion:
    def test_正常路径走通(self):
        r = FakeRetriever([make_hit(make_chunk())], detected=["pingan"])
        c = fake_client("归母净利润 134,778 百万元 [1]")
        res = answer_question("中国平安2025年归母净利润？", retriever=r,
                              llm_config=LLM_ON, llm_client=c)
        assert res.answered_by_llm and not res.refused
        assert res.answer == "归母净利润 134,778 百万元 [1]"
        assert res.cited_indexes == [1]
        assert res.suspicious_pages == []

    def test_检索层是被原样调用的(self):
        r = FakeRetriever([make_hit(make_chunk())], detected=["pingan"])
        answer_question("q", retriever=r, top_k=7, mode="bm25",
                        company="cpic", llm_config=LLM_ON,
                        llm_client=fake_client())
        call = r.search_calls[0]
        assert call["query"] == "q" and call["top_k"] == 7
        assert call["mode"] == "bm25" and call["company"] == "cpic"

    def test_一条都没召回就拒答(self):
        r = FakeRetriever([], cosine=0.0)
        res = answer_question("q", retriever=r, llm_config=LLM_ON,
                              llm_client=fake_client())
        assert res.refused and res.refusal_reason == "no_hits"
        assert not res.answered_by_llm
        assert "没有" in res.answer

    def test_相似度太低就拒答且不调用模型(self):
        r = FakeRetriever([make_hit(make_chunk())], cosine=0.30)
        c = fake_client()
        res = answer_question("红烧肉怎么做？", retriever=r,
                              llm_config=LLM_ON, llm_client=c)
        assert res.refused and res.refusal_reason == "similarity"
        assert c.calls == []                    # 关键：省 token，一行都不发
        assert res.sources                     # 但检索结果照样展示

    def test_年份对不上就拒答且不调用模型(self):
        r = FakeRetriever([make_hit(make_chunk(report_year=2025,
                                               text="本期净利润 100"))],
                          cosine=0.9, detected=["pingan"])
        c = fake_client()
        res = answer_question("中国平安2026年的净利润？", retriever=r,
                              llm_config=LLM_ON, llm_client=c)
        assert res.refused and res.refusal_reason == "year"
        assert "2026" in res.answer
        assert c.calls == []

    def test_没配大模型时不编答案(self):
        r = FakeRetriever([make_hit(make_chunk())], detected=["pingan"])
        res = answer_question("q", retriever=r,
                              llm_config=LLMConfig(provider="none"))
        assert res.refusal_reason == "no_llm"
        assert res.answer == ""                 # 不是模板假答案
        assert res.sources and res.materials    # 检索结果和引用照常给

    def test_调用模型失败不崩(self):
        r = FakeRetriever([make_hit(make_chunk())], detected=["pingan"])
        res = answer_question("q", retriever=r, llm_config=LLM_ON,
                              llm_client=fake_client(error=LLMError("网络炸了")))
        assert res.refused and res.refusal_reason == "error"
        assert "网络炸了" in " ".join(res.notes)

    def test_自定义阈值生效(self):
        r = FakeRetriever([make_hit(make_chunk())], cosine=0.50)
        res = answer_question("q", retriever=r, llm_config=LLM_ON,
                              llm_client=fake_client(), min_cosine=0.40)
        assert res.answered_by_llm               # 0.50 >= 0.40 放行

    def test_top_k控制材料条数(self):
        hits = [make_hit(make_chunk(cid=f"c{i}", page=i), rank=i)
                for i in range(1, 9)]
        r = FakeRetriever(hits, detected=["pingan"])
        res = answer_question("q", retriever=r, top_k=3, llm_config=LLM_ON,
                              llm_client=fake_client())
        assert len(res.materials) == 3 and len(res.sources) == 3

    def test_正文提到材料外的页码会加警告(self):
        r = FakeRetriever([make_hit(make_chunk(page=40))], detected=["pingan"])
        res = answer_question("q", retriever=r, llm_config=LLM_ON,
                              llm_client=fake_client("见第 999 页 [1]"))
        assert res.suspicious_pages == [999]
        assert any("⚠️" in n for n in res.notes)

    def test_没标注材料编号会加提醒(self):
        r = FakeRetriever([make_hit(make_chunk())], detected=["pingan"])
        res = answer_question("q", retriever=r, llm_config=LLM_ON,
                              llm_client=fake_client("净利润是 134,778 百万元"))
        assert res.cited_indexes == []
        assert any("没有标注" in n for n in res.notes)

    def test_材料截断参数传下去(self):
        r = FakeRetriever([make_hit(make_chunk(text="甲" * 3000))],
                          detected=["pingan"])
        res = answer_question("q", retriever=r, llm_config=LLM_ON,
                              llm_client=fake_client(),
                              max_material_chars=50)
        assert len(res.materials[0].text) < 200


# ------------------------------------------------- 防伪造：引用不由模型产生
class TestCitationsComeFromMetadata:
    """整个第三阶段最要紧的一条不变量。

    模型在正文里怎么胡说都行，页面上展示的引用块来自 chunk 元数据 ——
    所以引用永远不会被模型带偏。
    """

    def test_模型正文里的假页码不影响引用块(self):
        ch = make_chunk(company="中国平安", page=40, section="财务摘要")
        r = FakeRetriever([make_hit(ch)], detected=["pingan"])
        res = answer_question("q", retriever=r, llm_config=LLM_ON,
                              llm_client=fake_client(
                                  "根据第 999 页，净利润是 999,999 亿元 [1]"))
        # 引用块仍然是 chunk 里那一页
        assert res.sources[0].chunk.page == 40
        assert "页码：40页" in res.sources[0].chunk.citation()
        # 而模型编的页码被单独标出来，不静默改写
        assert res.suspicious_pages == [999]
        assert any("⚠️" in n for n in res.notes)

    def test_引用块直接用chunk的citation(self):
        ch = make_chunk(company="中国人寿", page=7, section="第二节 主要财务数据")
        r = FakeRetriever([make_hit(ch)], detected=[])
        res = answer_question("q", retriever=r, llm_config=LLM_ON,
                              llm_client=fake_client("随便写点什么"))
        assert res.sources[0].chunk.citation() == ch.citation()
        assert "公司：中国人寿" in ch.citation()
        assert "报告：2025年年度报告" in ch.citation()

    def test_送给模型的材料里带着正确的来源(self):
        ch = make_chunk(company="中国太保", page=88, section="合并资产负债表")
        r = FakeRetriever([make_hit(ch)], detected=[])
        c = fake_client()
        answer_question("q", retriever=r, llm_config=LLM_ON, llm_client=c)
        _, user = c.calls[0]
        assert "公司：中国太保" in user
        assert "页码：88页" in user

    def test_系统提示词要求不编造(self):
        r = FakeRetriever([make_hit(make_chunk())], detected=["pingan"])
        c = fake_client()
        answer_question("q", retriever=r, llm_config=LLM_ON, llm_client=c)
        system, _ = c.calls[0]
        assert "只使用材料里出现过的信息" in system
        assert "不许编造" in system


class TestQAResult:
    def test_拒答时has_answer为假(self):
        assert not QAResult(question="q", answer="有字", refused=True).has_answer

    def test_空答案has_answer为假(self):
        assert not QAResult(question="q", answer="").has_answer

    def test_正常答案has_answer为真(self):
        assert QAResult(question="q", answer="有字").has_answer
