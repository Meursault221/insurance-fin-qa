"""第二阶段检索层的测试：分词、BM25、融合、公司过滤。

这里全部用内存里的假 Chunk，不读 data/chunks —— 单元测试不该依赖
已经跑过第一阶段。真实数据上的效果由 scripts/eval_retrieval.py 衡量。
"""

from __future__ import annotations

import pytest

from src.bm25 import BM25Index
from src.chunker import Chunk
from src.hybrid import fuse, rrf_fuse, union_fuse, weighted_fuse
from src.retriever import Retriever, _key_of_doc, verify_consistency
from src import tokenize as tk


def make_chunk(cid: str, text: str, *, company: str = "中国平安",
               doc_id: str = "pingan_2025_annual", page: int = 1,
               section: str = "财务摘要") -> Chunk:
    return Chunk(chunk_id=cid, doc_id=doc_id, company=company,
                 report_year=2025, report_name="2025年年度报告",
                 page=page, section=section, text=text)


# ---------------------------------------------------------------- 分词
class TestTokenize:
    def test_千分位逗号被去掉(self):
        assert tk.tokenize("2,027,592") == ["2027592"]

    def test_带逗号和不带逗号切出来一样(self):
        assert tk.tokenize("1,234,567") == tk.tokenize("1234567")

    def test_数字不被中文分词器切碎(self):
        assert "2027592" in tk.tokenize("原保险保费收入2,027,592元")

    def test_小数和百分号保留(self):
        assert "7298.87" in tk.tokenize("总保费7,298.87亿元")
        assert "96.8%" in tk.tokenize("综合成本率96.8%")

    def test_中文词内补bigram(self):
        # jieba 把"归属于"切成一个词，bigram 再补出"归属""属于"，
        # 这样问"归属净利润"也能命中原文的"归属于母公司股东的净利润"
        toks = tk.tokenize("归属于")
        assert "归属" in toks and "属于" in toks

    def test_关掉bigram就没有字符bigram(self):
        toks = tk.tokenize("归属于", bigrams=False)
        assert toks == ["归属于"]

    def test_两字词不会被bigram重复计入(self):
        # 两字词的 bigram 就是它自己，补进去只会把 tf 翻倍
        toks = tk.tokenize("净利润")
        assert toks.count("净利润") == 1

    def test_停用词被剔除(self):
        for stop in ("的", "和", "公司", "集团"):
            assert stop not in tk.tokenize(f"{stop}净利润{stop}")

    def test_英文统一小写(self):
        assert "solvency" in tk.tokenize("Solvency Ratio")

    def test_空文本返回空(self):
        assert tk.tokenize("") == []

    def test_query与document用同一套规则(self):
        # 两边不一致的话 BM25 的 tf/idf 统计对不上，检索会静默变差
        s = "归属于母公司股东的净利润"
        assert tk.tokenize_query(s) == tk.tokenize(s)


class TestSimplifiedTraditional:
    """港股年报正文是繁体，占语料 52%。简体查询必须能命中繁体原文。

    不做简繁归一时，"中国人保的总资产" 和原文的 "集團總資產" 在字面上
    毫无交集 —— 这道最简单的关键词题会三路全挂。
    """

    PAIRS = [
        ("集團總資產20,275.92億元", "中国人保的总资产"),
        ("保險服務收入 103,087", "保险服务收入"),
        ("歸屬於母公司股東的淨利潤", "归属于母公司股东的净利润"),
        ("承保綜合成本率95.8%", "承保综合成本率"),
        ("原保險保費收入 555,777", "原保险保费收入"),
    ]

    def test_繁体文档与简体查询有共同token(self):
        for doc, q in self.PAIRS:
            inter = set(tk.tokenize(doc)) & set(tk.tokenize(q))
            assert inter, f"{q!r} 与 {doc!r} 没有任何共同 token"

    def test_同一个词繁简切出来一样(self):
        for trad, simp in [("總資產", "总资产"), ("保險服務收入", "保险服务收入"),
                           ("淨利潤", "净利润"), ("保費", "保费")]:
            assert tk.tokenize(trad) == tk.tokenize(simp), trad

    def test_转换不动数字(self):
        # 数字必须逐字不变 —— 这是整个项目的底线
        for s in ["總資產20,275.92億元", "保險服務收入 103,087 101,363 1.7",
                  "淨利潤9,771,143", "2025年12月31日，95.8%"]:
            digits = __import__("re").findall(r"[0-9][0-9,\.]*%?", s)
            assert __import__("re").findall(
                r"[0-9][0-9,\.]*%?", tk.to_simplified(s)) == digits, s

    def test_转换不动英文(self):
        s = "Management Review and Analysis 2025"
        assert tk.to_simplified(s) == s

    def test_简体文本转换后不变(self):
        s = "归属于母公司股东的净利润"
        assert tk.to_simplified(s) == s


# ---------------------------------------------------------------- BM25
class TestBM25:
    def _index(self):
        self.chunks = [
            make_chunk("c1", "归属于母公司股东的净利润134,778百万元", page=15),
            make_chunk("c2", "新业务价值36,897百万元", page=15),
            make_chunk("c3", "本公司总资产为人民币467亿元", page=10),
            make_chunk("c4", "归属于母公司股东的净利润154,078百万元",
                       company="中国人寿", doc_id="china_life_2025_annual", page=10),
        ]
        return BM25Index.build(self.chunks)

    def test_能按关键词召回(self):
        idx = self._index()
        hits = idx.search("归属于母公司股东的净利润", top_k=4)
        assert hits, "关键词检索不该返回空"
        assert hits[0][0] in {"c1", "c4"}

    def test_数字问题能命中带千分位的原文(self):
        idx = self._index()
        assert idx.search("134778", top_k=4)[0][0] == "c1"

    def test_全部doc的idf都是正数(self):
        # 加 1 平滑就是为了不让高频词变成负 idf
        idx = self._index()
        assert all(v > 0 for v in idx.idf.values())

    def test_词表里没有的词不报错只返回空(self):
        idx = self._index()
        assert idx.search("量子纠缠拓扑绝缘体", top_k=4) == []

    def test_空查询返回空(self):
        assert self._index().search("", top_k=4) == []

    def test_公司掩码在打分前生效(self):
        idx = self._index()
        allowed = [c.doc_id == "china_life_2025_annual" for c in self.chunks]
        hits = idx.search("归属于母公司股东的净利润", top_k=4, allowed=allowed)
        assert hits and all(cid == "c4" for cid, _ in hits)

    def test_掩码允许的doc一条都不匹配时返回空(self):
        idx = self._index()
        hits = idx.search("量子", top_k=4, allowed=[True, True, True, True])
        assert hits == []

    def test_exclude_ids被排除(self):
        idx = self._index()
        hits = idx.search("归属于母公司股东的净利润", top_k=4,
                          exclude_ids={"c1"})
        assert "c1" not in [cid for cid, _ in hits]

    def test_结果是降序的(self):
        idx = self._index()
        scores = [s for _, s in idx.search("净利润", top_k=4)]
        assert scores == sorted(scores, reverse=True)

    def test_落盘再读回结果一致(self, tmp_path):
        idx = self._index()
        idx.save(tmp_path / "bm25")
        back = BM25Index.load(tmp_path / "bm25")
        assert back.doc_ids == idx.doc_ids
        assert back.avgdl == idx.avgdl
        assert ([c for c, _ in back.search("净利润", top_k=4)]
                == [c for c, _ in idx.search("净利润", top_k=4)])

    def test_索引不存在时报错而不是静默返回空(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BM25Index.load(tmp_path / "nope")

    def test_stats字段齐全(self):
        st = self._index().stats()
        for k in ("docs", "terms", "postings", "avgdl", "min_len", "max_len"):
            assert k in st
        assert st["docs"] == 4


# ---------------------------------------------------------------- 融合
class TestFusion:
    def test_rrf只看名次不看分数(self):
        # 两路分数差 1000 倍，名次相同就该得到相同结果
        a = {"x": [("d1", 100.0), ("d2", 1.0)]}
        b = {"y": [("d1", 0.1), ("d2", 0.001)]}
        assert rrf_fuse(a) == rrf_fuse(b)

    def test_rrf两路都排第一的胜出(self):
        fused = rrf_fuse({"bm25": [("d1", 9.0), ("d2", 1.0)],
                          "vector": [("d1", 0.9), ("d3", 0.5)]})
        assert fused[0][0] == "d1"

    def test_rrf只有一路召回的也能进榜(self):
        fused = rrf_fuse({"bm25": [("d1", 9.0)], "vector": [("d2", 0.9)]})
        assert {c for c, _ in fused} == {"d1", "d2"}

    def test_rrf权重为0等于关掉这一路(self):
        rankings = {"bm25": [("d1", 1.0)], "vector": [("d2", 1.0)]}
        fused = dict(rrf_fuse(rankings, weights={"vector": 0}))
        assert "d2" not in fused
        assert "d1" in fused

    def test_rrf_top_k截断(self):
        rankings = {"a": [(f"d{i}", 1.0) for i in range(10)]}
        assert len(rrf_fuse(rankings, top_k=3)) == 3

    def test_rrf空输入返回空(self):
        assert rrf_fuse({}) == []

    def test_weighted归一化后加权(self):
        fused = dict(weighted_fuse({"bm25": [("d1", 10.0), ("d2", 0.0)],
                                    "vector": [("d2", 1.0), ("d1", 0.0)]},
                                   weights={"bm25": 1.0, "vector": 0.0}))
        assert "d1" in fused and fused["d1"] > fused.get("d2", 0)

    def test_weighted同分不会除零(self):
        out = weighted_fuse({"a": [("d1", 5.0), ("d2", 5.0)]})
        assert {c for c, _ in out} == {"d1", "d2"}

    def test_fuse按名字分发(self):
        rankings = {"bm25": [("d1", 1.0)]}
        assert fuse(rankings, method="rrf") == rrf_fuse(rankings)
        assert fuse(rankings, method="weighted") == weighted_fuse(rankings)
        assert fuse(rankings, method="union", top_k=5) == union_fuse(
            rankings, top_k=5, reserve=None)

    def test_未知融合方式报错(self):
        with pytest.raises(ValueError, match="未知融合方式"):
            fuse({}, method="magic")


class TestUnionFuse:
    """候选集限定为两路前 top_k 的并集。默认不用，见 src/hybrid.py 的说明。"""

    def test_候选集外的结果被剔除(self):
        # d9 只有 vector 一路召回，且排在很后面 —— 不该出现在结果里
        rankings = {"bm25": [(f"b{i}", 1.0) for i in range(10)],
                    "vector": [(f"v{i}", 1.0) for i in range(10)] + [("d9", 0.1)]}
        out = {c for c, _ in union_fuse(rankings, top_k=10, reserve=10)}
        assert "d9" not in out

    def test_两路前reserve名都保留(self):
        rankings = {"bm25": [(f"b{i}", 1.0) for i in range(10)],
                    "vector": [(f"v{i}", 1.0) for i in range(10)]}
        out = {c for c, _ in union_fuse(rankings, top_k=20, reserve=10)}
        assert out == {f"b{i}" for i in range(10)} | {f"v{i}" for i in range(10)}

    def test_并集内仍按rrf排序(self):
        # d1 两路都排第一，应当排在只有一路召回的前面
        rankings = {"bm25": [("d1", 9.0), ("d2", 1.0)],
                    "vector": [("d1", 0.9), ("d3", 0.5)]}
        out = [c for c, _ in union_fuse(rankings, top_k=3, reserve=3)]
        assert out[0] == "d1"

    def test_候选集只限外层名次不改变打分(self):
        # union 只砍候选，不重算分数：同一 chunk 的分数应与 rrf 一致
        rankings = {"bm25": [("d1", 9.0), ("d2", 1.0)],
                    "vector": [("d3", 0.9)]}
        u = dict(union_fuse(rankings, top_k=5, reserve=5))
        r = dict(rrf_fuse(rankings))
        assert u["d1"] == r["d1"]


# ---------------------------------------------------------------- 公司归属
class TestDocKey:
    KEYS = ["picc_pc", "new_china_life", "china_taiping", "china_re",
            "china_life", "sunshine", "zhongan", "pingan", "cpic", "picc"]

    def test_普通key(self):
        assert _key_of_doc("pingan_2025_annual", self.KEYS) == "pingan"

    def test_picc_pc不会被判成picc(self):
        # picc 是 picc_pc 的前缀。按短的先匹配会把中国财险的 chunk
        # 全判给中国人保 —— 两家公司的答案会互相串。
        assert _key_of_doc("picc_pc_2025_annual", self.KEYS) == "picc_pc"

    def test_picc本身仍然正确(self):
        assert _key_of_doc("picc_2025_annual", self.KEYS) == "picc"

    def test_带下划线的长key(self):
        for k in ("new_china_life", "china_taiping", "china_re", "china_life"):
            assert _key_of_doc(f"{k}_2025_annual", self.KEYS) == k


# ---------------------------------------------------------------- Retriever
class FakeVector:
    """向量索引替身。

    装 sentence-transformers 要拉 torch，不能让单元测试卡在这上面。
    这个替身只保证接口一致（doc_ids / search / stats），用来测
    「混合检索如何把两路合起来」这一层逻辑 —— 真实语义效果由
    scripts/eval_retrieval.py 在真索引上衡量。
    """

    def __init__(self, doc_ids, order=None, model_name="fake", dim=4):
        self.doc_ids = list(doc_ids)
        self.order = list(order or doc_ids)
        self.model_name = model_name
        self.dim = dim

    def search(self, query, top_k=10, *, allowed=None, exclude_ids=None):
        out = []
        for i, cid in enumerate(self.order):
            if cid not in self.doc_ids:
                continue
            if allowed is not None and not allowed[self.doc_ids.index(cid)]:
                continue
            if exclude_ids and cid in exclude_ids:
                continue
            out.append((cid, 1.0 - i * 0.01))
        return out[:top_k]

    def stats(self):
        return {"docs": len(self.doc_ids), "dim": self.dim,
                "model": self.model_name}


def _sample_chunks():
    return [
        make_chunk("p1", "归属于母公司股东的净利润134,778百万元",
                   page=15, section="财务摘要"),
        make_chunk("p2", "本集团总资产13,898,471百万元", page=44,
                   section="集团经营业绩"),
        make_chunk("l1", "归属于母公司股东的净利润154,078百万元",
                   company="中国人寿", doc_id="china_life_2025_annual",
                   page=10, section="财务摘要"),
        make_chunk("c1", "保险服务收入2,889.10亿元",
                   company="中国太保", doc_id="cpic_2025_annual",
                   page=36, section="经营情况讨论与分析"),
        make_chunk("pc1", "保险服务收入511,594百万元",
                   company="中国财险", doc_id="picc_pc_2025_annual",
                   page=4, section="财务摘要"),
        make_chunk("i1", "保险服务收入570,717百万元",
                   company="中国人保", doc_id="picc_2025_annual",
                   page=27, section="财务摘要"),
    ]


class TestRetriever:
    @pytest.fixture
    def retriever(self):
        """只有 BM25 的检索器。"""
        chunks = _sample_chunks()
        return Retriever(chunks, bm25=BM25Index.build(chunks))

    @pytest.fixture
    def hybrid(self):
        """BM25 + 向量替身，走完整的混合路径。"""
        chunks = _sample_chunks()
        return Retriever(chunks, bm25=BM25Index.build(chunks),
                         vector=FakeVector([c.chunk_id for c in chunks]))

    # ---- 公司识别
    def test_识别全称(self, retriever):
        assert retriever.detect_company("中国平安的总资产是多少") == "pingan"

    def test_识别简称(self, retriever):
        assert retriever.detect_company("平安的总资产") == "pingan"

    def test_识别不出公司返回None(self, retriever):
        assert retriever.detect_company("保险行业2025年发展情况") is None

    def test_多公司时detect_company返回None(self, retriever):
        # 对比题不能只挑一家 —— 硬过滤会把另一半答案切掉
        q = "比较中国平安和中国人寿的净利润"
        assert set(retriever.detect_companies(q)) == {"pingan", "china_life"}
        assert retriever.detect_company(q) is None

    def test_多公司默认每家公司都召回(self, retriever):
        # 全局 top-k 会被内容多的那家占满，另一家一条都进不来
        out = retriever.search("比较中国平安和中国人寿的净利润", top_k=4,
                               mode="bm25")
        got = {r.company for r in out}
        assert got == {"中国平安", "中国人寿"}, got

    def test_多公司可以退回全局召回(self, retriever):
        out = retriever.search("比较中国平安和中国人寿的净利润", top_k=4,
                               mode="bm25", multi_company="global")
        assert out
        assert all(r.source == "bm25" for r in out)

    def test_人保财险判给中国财险而不是中国人保(self, retriever):
        assert retriever.detect_company("人保财险的保险服务收入") == "picc_pc"

    def test_中国人保判给自己(self, retriever):
        assert retriever.detect_company("中国人保的保险服务收入") == "picc"

    # ---- 检索 + 过滤
    def test_按公司过滤后只剩该公司(self, retriever):
        out = retriever.search("归属于母公司股东的净利润", top_k=5,
                               company="china_life", mode="bm25")
        assert out and all(r.company == "中国人寿" for r in out)

    def test_自动识别公司并过滤(self, retriever):
        out = retriever.search("中国平安的总资产", top_k=5, mode="bm25")
        assert out and all(r.company == "中国平安" for r in out)

    def test_关闭自动识别则不按公司过滤(self, retriever):
        out = retriever.search("归属于母公司股东的净利润", top_k=5,
                               auto_company=False, mode="bm25")
        assert len({r.company for r in out}) > 1

    def test_结果带完整来源信息(self, retriever):
        r = retriever.search("中国平安的总资产", top_k=1, mode="bm25")[0]
        assert r.company == "中国平安"
        assert r.page == 44
        assert r.chunk.section == "集团经营业绩"
        assert r.company_key == "pingan"

    def test_引用格式符合作业要求(self, retriever):
        r = retriever.search("中国平安的总资产", top_k=1, mode="bm25")[0]
        c = r.citation()
        assert "公司：中国平安" in c
        assert "报告：2025年年度报告" in c
        assert "页码：44页" in c

    def test_排名从1连续(self, retriever):
        out = retriever.search("净利润", top_k=5, auto_company=False,
                               mode="bm25")
        assert [r.rank for r in out] == list(range(1, len(out) + 1))

    def test_未知公司key报错(self, retriever):
        with pytest.raises(KeyError, match="可用 key"):
            retriever.search("净利润", company="不存在公司", mode="bm25")

    def test_未知模式报错(self, retriever):
        with pytest.raises(ValueError, match="未知检索模式"):
            retriever.search("净利润", mode="telepathy")

    def test_没有向量索引时用向量模式报错(self, retriever):
        # 默认的 hybrid 模式也依赖向量索引，缺了必须报错而不是悄悄退化成 BM25
        with pytest.raises(RuntimeError, match="向量索引"):
            retriever.search("净利润", mode="vector")
        with pytest.raises(RuntimeError, match="向量索引"):
            retriever.search("净利润")

    def test_bm25模式只走BM25(self, retriever):
        out = retriever.search("净利润", top_k=3, mode="bm25",
                               auto_company=False)
        assert out and all(r.source == "bm25" for r in out)

    # ---- 混合检索
    def test_search_per_source返回三路(self, hybrid):
        got = hybrid.search_per_source("净利润", top_k=3)
        assert set(got) == {"bm25", "vector", "hybrid"}
        assert got["bm25"] and got["vector"] and got["hybrid"]

    def test_混合结果含两路都认可的那条(self, hybrid):
        # FakeVector 按 doc_ids 原序给分，BM25 的最优是 p1 —— 两路都排前的
        # 应该出现在融合结果里
        out = hybrid.search("归属于母公司股东的净利润", top_k=6,
                            auto_company=False, mode="hybrid")
        assert "p1" in {r.chunk_id for r in out}

    def test_混合模式只召回向量能召回的那条(self):
        # 造一个只有向量"看得懂"的查询：BM25 一个词都匹配不上
        chunks = _sample_chunks()
        v = FakeVector([c.chunk_id for c in chunks], order=["c1", "p1", "p2"])
        r = Retriever(chunks, bm25=BM25Index.build(chunks), vector=v)
        out = r.search("量子纠缠", top_k=3, auto_company=False, mode="hybrid")
        assert [x.chunk_id for x in out] == ["c1", "p1", "p2"]
        # 而 BM25 单独跑这条查询是空的 —— 说明混合确实把向量那一路用上了
        assert r.search("量子纠缠", top_k=3, auto_company=False,
                        mode="bm25") == []

    def test_混合模式下公司过滤对两路都生效(self, hybrid):
        out = hybrid.search("中国太保的保险服务收入", top_k=5, mode="hybrid")
        assert out and all(r.company == "中国太保" for r in out)

    def test_stats统计每家chunk数(self, retriever):
        st = retriever.stats()
        assert st["chunks"] == 6
        assert st["companies"] == 5
        assert st["by_company"]["pingan"] == 2

    # ---- 一致性校验
    def test_索引与chunks不匹配时报错(self):
        chunks = [make_chunk("p1", "净利润", page=1)]
        idx = BM25Index.build([make_chunk("别的", "净利润", page=9)])
        with pytest.raises(ValueError, match="不一致"):
            verify_consistency(chunks, {"BM25": idx})

    def test_索引条数对但顺序不同也要报错(self):
        # 顺序不同最阴：分数是对的，指向的 chunk 却是另一个
        chunks = [make_chunk("a", "净利润", page=1),
                  make_chunk("b", "净利润", page=2)]
        idx = BM25Index.build([make_chunk("b", "净利润", page=2),
                               make_chunk("a", "净利润", page=1)])
        with pytest.raises(ValueError, match="不一致"):
            verify_consistency(chunks, {"BM25": idx})

    def test_一致时不报错(self):
        chunks = [make_chunk("a", "净利润", page=1)]
        assert verify_consistency(
            chunks, {"BM25": BM25Index.build(chunks)}) is None

    def test_索引为None时跳过校验(self):
        assert verify_consistency([make_chunk("a", "x", page=1)],
                                  {"向量": None}) is None
