"""切分逻辑测试（手工构造 ExtractedDocument，不需要 PDF）。"""

from src.chunker import ChunkConfig, chunk_document
from src.schema import (DocumentMeta, ExtractedDocument, PageRecord, TableBlock,
                        TextBlock)


def meta() -> DocumentMeta:
    return DocumentMeta(
        doc_id="demo_2025_annual", company="示例公司", company_full="",
        company_key="demo", stock_codes=["000000.SH"], exchange="",
        report_year=2025, report_name="2025年年度报告", report_type="annual",
        source_file="demo.pdf", source_path="data/raw/demo.pdf",
    )


def blk(i, text, kind="paragraph", level=0, section="", y=0.0, inside_table=0):
    return TextBlock(index=i, page=1, bbox=[0, y, 500, y + 12], text=text,
                     kind=kind, level=level, section=section,
                     inside_table=inside_table)


def table(i, page=1, section="", y=200.0):
    cells = [["项目", "2025年"], ["营业收入", "1,234"]]
    return TableBlock(index=i, page=page, bbox=[0, y, 500, y + 60],
                      n_rows=2, n_cols=2, cells=cells,
                      header=["项目", "2025年"],
                      text="| 项目 | 2025年 |\n| --- | --- |\n| 营业收入 | 1,234 |",
                      section=section)


def doc_with(pages):
    return ExtractedDocument(meta=meta(), pages=pages)


class TestBasic:
    def test_每个chunk都带来源五要素(self):
        page = PageRecord(page=7, blocks=[
            blk(1, "第三节 管理层讨论与分析", kind="heading", level=1,
                section="第三节 管理层讨论与分析"),
            blk(2, "本集团2025年实现营业收入人民币1,234亿元。" * 3, y=30),
        ])
        chunks = chunk_document(doc_with([page]))
        assert chunks
        for c in chunks:
            assert c.company == "示例公司"
            assert c.report_year == 2025
            assert c.report_name == "2025年年度报告"
            assert c.page == 7
            assert c.section

    def test_章节被继承(self):
        page = PageRecord(page=3, blocks=[
            blk(1, "第二节 主要财务数据", kind="heading", level=1,
                section="第二节 主要财务数据"),
            blk(2, "下面这段文字应该继承上一行的章节路径。" * 4, y=30),
        ])
        chunks = chunk_document(doc_with([page]))
        assert any("第二节 主要财务数据" in c.section for c in chunks)

    def test_页码归属正确(self):
        pages = [
            PageRecord(page=1, blocks=[blk(1, "第一页的内容。" * 10)]),
            PageRecord(page=2, blocks=[blk(1, "第二页的内容。" * 10)]),
        ]
        chunks = chunk_document(doc_with(pages))
        assert {c.page for c in chunks} == {1, 2}
        assert all(c.page_end == c.page for c in chunks)


class TestTableAtomicity:
    def test_表格是独立chunk(self):
        page = PageRecord(page=5, blocks=[blk(1, "表格前面的一段话。" * 5)],
                          tables=[table(1, page=5, section="第二节")])
        chunks = chunk_document(doc_with([page]))
        tbl_chunks = [c for c in chunks if c.block_type == "table"]
        assert len(tbl_chunks) == 1
        assert "1,234" in tbl_chunks[0].text

    def test_表格单元格随chunk一起保存(self):
        page = PageRecord(page=5, blocks=[], tables=[table(1, page=5)])
        chunks = chunk_document(doc_with([page]))
        tc = [c for c in chunks if c.block_type == "table"][0]
        assert tc.tables[0]["cells"][1] == ["营业收入", "1,234"]

    def test_超长表格不被切断(self):
        cells = [["列A", "列B"]] + [[f"行{i}", str(i * 111)] for i in range(400)]
        big = TableBlock(
            index=1, page=2, bbox=[0, 0, 500, 500], n_rows=len(cells), n_cols=2,
            cells=cells, header=["列A", "列B"],
            text="| 列A | 列B |\n| --- | --- |\n" + "\n".join(
                f"| {r[0]} | {r[1]} |" for r in cells[1:]),
        )
        page = PageRecord(page=2, blocks=[], tables=[big])
        chunks = chunk_document(doc_with([page]))
        tc = [c for c in chunks if c.block_type == "table"]
        assert len(tc) == 1, "表格被切开了"
        assert "行399" in tc[0].text

    def test_关闭表格时只有文本(self):
        page = PageRecord(page=2, blocks=[blk(1, "正文内容。" * 5)], tables=[table(1)])
        chunks = chunk_document(doc_with([page]), ChunkConfig(keep_tables=False))
        assert all(c.block_type != "table" for c in chunks)


class TestTableTextDedup:
    def test_默认不重复收录表格区域文本(self):
        page = PageRecord(
            page=1,
            blocks=[blk(1, "| 项目 | 2025年 |", kind="table_text", inside_table=1),
                    blk(2, "真正的正文段落。" * 5, y=300)],
            tables=[table(1)],
        )
        chunks = chunk_document(doc_with([page]))
        non_table = [c for c in chunks if c.block_type != "table"]
        assert all("2025年 |" not in c.text for c in non_table)

    def test_可以打开开关收录(self):
        page = PageRecord(
            page=1,
            blocks=[blk(1, "| 项目 | 2025年 |", kind="table_text", inside_table=1)],
            tables=[],
        )
        chunks = chunk_document(doc_with([page]),
                                ChunkConfig(include_table_text=True))
        assert chunks and "2025年" in chunks[0].text


class TestSplitting:
    def test_长文本被切成多块(self):
        long_text = "这是一句测试用的中文句子，用来凑长度。" * 200
        page = PageRecord(page=1, blocks=[blk(1, long_text)])
        chunks = chunk_document(doc_with([page]), ChunkConfig(max_chars=300))
        assert len(chunks) > 1
        assert all(c.n_chars <= 400 for c in chunks)

    def test_切分不去掉任何字符总数(self):
        long_text = "甲" * 2000
        page = PageRecord(page=1, blocks=[blk(1, long_text)])
        cfg = ChunkConfig(max_chars=300, overlap_chars=0, min_chars=0)
        chunks = chunk_document(doc_with([page]), cfg)
        assert sum(c.n_chars for c in chunks) == 2000

    def test_短碎片被合并(self):
        page = PageRecord(page=1, blocks=[
            blk(1, "主段落内容。" * 30, y=0),
            blk(2, "尾巴", y=100),
        ])
        chunks = chunk_document(doc_with([page]), ChunkConfig(min_chars=120))
        assert len(chunks) == 1
        assert "尾巴" in chunks[0].text

    def test_空文档返回空列表(self):
        assert chunk_document(doc_with([PageRecord(page=1)])) == []


class TestIds:
    def test_chunk_id唯一且含页码(self):
        pages = [PageRecord(page=p, blocks=[blk(1, f"第{p}页内容。" * 10)])
                 for p in (1, 2, 3)]
        chunks = chunk_document(doc_with(pages))
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
        assert any("_p0002_" in i for i in ids)

    def test_chunk_index连续(self):
        pages = [PageRecord(page=p, blocks=[blk(1, f"第{p}页。" * 20)])
                 for p in (1, 2, 3)]
        chunks = chunk_document(doc_with(pages))
        assert [c.chunk_index for c in chunks] == list(range(1, len(chunks) + 1))

    def test_context_prefix含公司报告页码(self):
        page = PageRecord(page=9, blocks=[blk(1, "正文。" * 20)])
        chunks = chunk_document(doc_with([page]))
        assert "示例公司" in chunks[0].context_prefix
        assert "2025年年度报告" in chunks[0].context_prefix
        assert "第9页" in chunks[0].context_prefix
