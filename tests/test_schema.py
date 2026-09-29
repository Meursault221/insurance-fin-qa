"""数据结构序列化 / 反序列化测试。"""

from src.schema import (Chunk, DocumentMeta, ExtractedDocument, PageRecord,
                        TableBlock, TextBlock)


def make_meta() -> DocumentMeta:
    return DocumentMeta(
        doc_id="demo_2025_annual",
        company="示例公司",
        company_full="示例公司股份有限公司",
        company_key="demo",
        stock_codes=["000000.SH"],
        exchange="示例交易所",
        report_year=2025,
        report_name="2025年年度报告",
        report_type="annual",
        source_file="demo.pdf",
        source_path="data/raw/demo.pdf",
    )


class TestDocumentMeta:
    def test_roundtrip(self):
        m = make_meta()
        m2 = DocumentMeta.from_dict(m.to_dict())
        assert m2 == m

    def test_必填来源字段齐全(self):
        m = make_meta()
        d = m.to_dict()
        for k in ("company", "report_year", "report_name"):
            assert k in d


class TestChunk:
    def test_来源五要素齐全(self):
        c = Chunk(
            chunk_id="x_p0001_c0001", doc_id="x",
            company="中国平安", report_year=2025,
            report_name="2025年年度报告", page=12,
            section="第三节 管理层讨论与分析", text="……",
        )
        d = c.to_dict()
        for k in ("company", "report_year", "report_name", "page", "section", "text"):
            assert k in d, f"缺少来源字段 {k}"

    def test_citation_包含作业要求的四项(self):
        c = Chunk(chunk_id="x", doc_id="x", company="中国平安", report_year=2025,
                  report_name="2025年年度报告", page=12,
                  section="第三节 管理层讨论与分析", text="t")
        s = c.citation()
        assert "公司：中国平安" in s
        assert "报告：2025年年度报告" in s
        assert "页码：12页" in s
        assert "第三节 管理层讨论与分析" in s

    def test_embed_text_带上下文前缀(self):
        c = Chunk(chunk_id="x", doc_id="x", company="A", report_year=2025,
                  report_name="R", page=1, section="S", text="正文",
                  context_prefix="【A·R】第1页 S")
        assert c.embed_text.startswith("【A·R】")
        assert c.embed_text.endswith("正文")
        assert c.text == "正文"      # 展示用的 text 不带前缀

    def test_roundtrip(self):
        c = Chunk(chunk_id="x", doc_id="x", company="A", report_year=2025,
                  report_name="R", page=1, section="S", text="正文",
                  tables=[{"index": 1, "cells": [["a", "1"]]}])
        c2 = Chunk.from_dict(c.to_dict())
        assert c2.company == "A"
        assert c2.tables[0]["cells"] == [["a", "1"]]


class TestNested:
    def test_完整文档roundtrip(self):
        meta = make_meta()
        tab = TableBlock(index=1, page=1, n_rows=2, n_cols=2,
                         cells=[["项目", "金额"], ["A", "1,234"]],
                         text="| 项目 | 金额 |\n| --- | --- |\n| A | 1,234 |")
        blk = TextBlock(index=1, page=1, text="一、主要财务数据",
                        kind="heading", level=2, section="一、主要财务数据")
        page = PageRecord(page=1, blocks=[blk], tables=[tab], text="一、主要财务数据")
        doc = ExtractedDocument(meta=meta, pages=[page], extractor="PyMuPDF")

        d = doc.to_dict()
        doc2 = ExtractedDocument.from_dict(d)

        assert doc2.meta.company == "示例公司"
        assert len(doc2.pages) == 1
        assert doc2.pages[0].blocks[0].kind == "heading"
        assert doc2.pages[0].tables[0].cells[1][1] == "1,234"

    def test_多余字段被忽略(self):
        m = DocumentMeta.from_dict({
            "doc_id": "a", "company": "b", "company_full": "", "company_key": "b",
            "stock_codes": [], "exchange": "", "report_year": 2025,
            "report_name": "r", "report_type": "annual",
            "source_file": "f", "source_path": "p",
            "某个未来才加的字段": 123,
        })
        assert m.company == "b"
