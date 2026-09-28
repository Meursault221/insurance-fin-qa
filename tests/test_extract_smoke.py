"""端到端冒烟测试：合成 PDF -> 抽取 -> 切分。

没有 PyMuPDF 或 reportlab 时自动跳过（不会让整个测试套挂掉）。
合成样本由 scripts/make_demo_pdf.py 生成，不含任何真实财务数据。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import chunker, config, pdf_extractor, pymupdf_compat  # noqa: E402
from src.chunker import ChunkConfig  # noqa: E402

pytestmark = pytest.mark.integration

DEMO_PDF = config.FIXTURES_DIR / "demo_company_2025_annual.pdf"


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


requires = pytest.mark.skipif(
    not (pymupdf_compat.is_available() and _has("reportlab")),
    reason="需要 PyMuPDF 和 reportlab",
)


@pytest.fixture(scope="module")
def demo_doc():
    if not DEMO_PDF.exists():
        sys.path.insert(0, str(ROOT / "scripts"))
        from make_demo_pdf import build
        build(DEMO_PDF)
    meta = pdf_extractor.build_meta(
        DEMO_PDF,
        company="示例公司", company_key="demo", report_year=2025,
        report_name="2025年年度报告（合成样本）", report_type="annual",
    )
    return pdf_extractor.extract_pdf(DEMO_PDF, meta, table_backend="pymupdf")


@requires
class TestExtraction:
    def test_页数(self, demo_doc):
        assert demo_doc.meta.page_count == 2

    def test_有文本(self, demo_doc):
        assert demo_doc.stats["char_count"] > 100

    def test_每页都有页码字段(self, demo_doc):
        for p in demo_doc.pages:
            assert p.page >= 1

    def test_识别到印刷页码(self, demo_doc):
        printed = [p.printed_page for p in demo_doc.pages]
        assert any(x is not None for x in printed), f"没识别到页脚页码: {printed}"

    def test_识别到标题(self, demo_doc):
        headings = [b.text for _p, b in demo_doc.iter_blocks() if b.kind == "heading"]
        assert any("重要提示" in h or "主要财务数据" in h for h in headings), headings

    def test_章节路径被填充(self, demo_doc):
        sections = {b.section for _p, b in demo_doc.iter_blocks() if b.section}
        assert sections, "所有 block 的 section 都是空的"

    def test_抽到表格(self, demo_doc):
        tables = [t for _p, t in demo_doc.iter_tables()]
        assert tables, "没抽到任何表格"
        assert all(t.n_rows >= 2 and t.n_cols >= 2 for t in tables)

    def test_表格数字完整(self, demo_doc):
        """合成样本里的占位数字必须原样出现在表格文本中。"""
        all_text = "\n".join(t.text for _p, t in demo_doc.iter_tables())
        for n in ["1,234", "12,345,678", "8,888", "13,332"]:
            assert n in all_text, f"表格里丢了数字 {n}"

    def test_表格结构行列对齐(self, demo_doc):
        for _p, t in demo_doc.iter_tables():
            widths = {len(r) for r in t.cells}
            assert len(widths) == 1, f"表格 {t.index} 行列错位: {widths}"

    def test_页码归属正确(self, demo_doc):
        """第 2 页的表格，page 字段必须等于 2。"""
        for p, t in demo_doc.iter_tables():
            assert t.page == p.page

    def test_表格章节按位置归属(self, demo_doc):
        """第 1 页的表在"第二节"下，不能被赋成该页最后一节的标题。"""
        p1 = demo_doc.pages[0]
        assert p1.tables
        for t in p1.tables:
            assert "第二节" in t.section, f"表格章节错了: {t.section!r}"

        p2 = demo_doc.pages[1]
        for t in p2.tables:
            assert "（一）示例业务分部情况" in t.section, f"表格章节错了: {t.section!r}"
            assert "三、示例风险提示" not in t.section, "用到了整页跑完后的 tracker 状态"

    def test_页脚页码不算正文块(self, demo_doc):
        for p in demo_doc.pages:
            footers = [b for b in p.blocks if b.kind == "page_footer"]
            assert footers, f"第 {p.page} 页没识别出页脚"
            body_text = "\n".join(b.text for b in p.blocks
                                  if b.kind not in ("table_text", "page_footer"))
            assert not body_text.rstrip().endswith("\n" + str(p.page))

    def test_表格题注不是正文段落(self, demo_doc):
        t1 = demo_doc.pages[0].tables[0]
        assert "单位" in t1.caption, f"题注抓错了: {t1.caption!r}"


@requires
class TestChunking:
    def test_chunk都带完整来源(self, demo_doc):
        chunks = chunker.chunk_document(demo_doc, ChunkConfig())
        assert chunks
        for c in chunks:
            assert c.company == "示例公司"
            assert c.report_year == 2025
            assert "2025年年度报告" in c.report_name
            assert c.page >= 1

    def test_citation格式(self, demo_doc):
        chunks = chunker.chunk_document(demo_doc, ChunkConfig())
        s = chunks[0].citation()
        assert "公司：" in s and "报告：" in s and "页码：" in s

    def test_表格chunk存在且数字在(self, demo_doc):
        chunks = chunker.chunk_document(demo_doc, ChunkConfig())
        tc = [c for c in chunks if c.block_type == "table"]
        assert tc
        assert any("1,234" in c.text or "8,888" in c.text for c in tc)

    def test_落盘再读回一致(self, demo_doc, tmp_path):
        chunks = chunker.chunk_document(demo_doc, ChunkConfig())
        p = tmp_path / "c.jsonl"
        n = chunker.save_chunks(chunks, p)
        assert n == len(chunks)
        back = chunker.load_chunks(p)
        assert len(back) == len(chunks)
        assert back[0].text == chunks[0].text
        assert back[0].citation() == chunks[0].citation()

    def test_JSON序列化再读回一致(self, demo_doc, tmp_path):
        p = pdf_extractor.save_extracted(demo_doc, tmp_path)
        doc2 = pdf_extractor.load_extracted(p)
        assert doc2.meta.company == demo_doc.meta.company
        assert len(doc2.pages) == len(demo_doc.pages)
        t1 = list(demo_doc.iter_tables())[0][1]
        t2 = list(doc2.iter_tables())[0][1]
        assert t1.cells == t2.cells


@requires
class TestPdfplumberBackend:
    @pytest.mark.skipif(not _has("pdfplumber"), reason="未安装 pdfplumber")
    def test_后端可切换(self):
        if not DEMO_PDF.exists():
            pytest.skip("演示 PDF 未生成")
        meta = pdf_extractor.build_meta(
            DEMO_PDF, company="示例公司", company_key="demo",
            report_year=2025, report_name="2025年年度报告（合成样本）",
        )
        doc = pdf_extractor.extract_pdf(DEMO_PDF, meta, table_backend="pdfplumber")
        assert doc.table_backend == "pdfplumber"
        assert doc.stats["char_count"] > 100


class TestJoinSpans:
    """同行内的 span 拼接：相邻财务数字不能被粘成一个数。"""

    @staticmethod
    def _spans(*texts):
        return [{"text": t} for t in texts]

    def test_相邻千分位数字不粘连(self):
        from src.pdf_extractor import _join_spans
        out = _join_spans(self._spans("376,112", "1,304,712"))
        assert out == "376,112 1,304,712", out

    def test_反向粘连也不漏(self):
        from src.pdf_extractor import _join_spans
        out = _join_spans(self._spans("1,234", "5,678"))
        assert out == "1,234 5,678", out

    def test_普通文字不被塞空格(self):
        from src.pdf_extractor import _join_spans
        assert _join_spans(self._spans("营业", "收入")) == "营业收入"

    def test_括号负数不被拆开(self):
        from src.pdf_extractor import _join_spans
        assert _join_spans(self._spans("(1", ",234)")) == "(1,234)"

    def test_单个span原样返回(self):
        from src.pdf_extractor import _join_spans
        assert _join_spans(self._spans("1,234.56")) == "1,234.56"

    def test_空输入(self):
        from src.pdf_extractor import _join_spans
        assert _join_spans([]) == ""
