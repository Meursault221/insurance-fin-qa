"""表格线性化 / 单元格清洗的单元测试。

重点验证一件事：**数字不能被改动**。
"""

from src import textnorm
from src.textnorm import (clean_cell, detect_merged_cells, is_number_like,
                          linearize_table, normalize, table_to_records)


class TestCleanCell:
    def test_None变空串不是0(self):
        assert clean_cell(None) == ""

    def test_数字原样保留千分位(self):
        assert clean_cell("1,234,567.89") == "1,234,567.89"

    def test_负数括号原样保留(self):
        assert clean_cell("(1,234)") == "(1,234)"

    def test_破折号不被改成0(self):
        for dash in ["—", "–", "-", "－"]:
            assert clean_cell(dash) == dash

    def test_全角空格和NBSP被规范化(self):
        assert clean_cell("营业收入　  1,234") == "营业收入 1,234"

    def test_单元格内换行变空格(self):
        assert clean_cell("营业收入\n1,234") == "营业收入 1,234"

    def test_百分数保留(self):
        assert clean_cell("11.10%") == "11.10%"


class TestNormalize:
    def test_删除零宽字符(self):
        assert normalize("营​业‌收入") == "营业收入"

    def test_保留换行(self):
        assert "\n" in normalize("第一行\n第二行")

    def test_压缩行内多空格(self):
        assert normalize("A     B") == "A B"


class TestLinearize:
    def test_基本结构(self):
        cells = [["项目", "2025年"], ["营业收入", "1,234"]]
        out = linearize_table(cells)
        lines = out.split("\n")
        assert lines[0] == "| 项目 | 2025年 |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| 营业收入 | 1,234 |"

    def test_数字一个不丢(self):
        cells = [["a", "1,234,567"], ["b", "9,999.99"], ["c", "(1,000)"]]
        out = linearize_table(cells)
        for n in ["1,234,567", "9,999.99", "(1,000)"]:
            assert n in out

    def test_None渲染成空格而不是0(self):
        cells = [["a", None, "1"], ["b", "2", None]]
        out = linearize_table(cells)
        assert "0" not in out.replace("| ---", "")  # 没有凭空出现的 0
        assert out.count("|") > 0

    def test_行宽不齐自动补齐(self):
        cells = [["a", "b", "c"], ["1"]]
        out = linearize_table(cells)
        rows = [l for l in out.split("\n") if l and not l.startswith("| ---")]
        widths = {len(r.split("|")) for r in rows}
        assert len(widths) == 1, "各行管道数应一致"

    def test_单元格里的竖线被转义(self):
        out = linearize_table([["a|b", "c"]])
        assert "\\|" in out

    def test_带表头参数(self):
        out = linearize_table([["1", "2"]], header=["甲", "乙"])
        assert out.split("\n")[0] == "| 甲 | 乙 |"

    def test_空表返回空串(self):
        assert linearize_table([]) == ""

    # ---- 回归：extract() 的 cells 一般含表头行，不能渲染两遍 ----
    def test_表头行不重复(self):
        cells = [["项目", "2025年"], ["营业收入", "1,234"]]
        out = linearize_table(cells, header=["项目", "2025年"])
        data_rows = [l for l in out.split("\n") if l and not l.startswith("| ---")]
        assert len(data_rows) == 2, f"表头重复了：\n{out}"
        assert data_rows[0] == "| 项目 | 2025年 |"
        assert data_rows[1] == "| 营业收入 | 1,234 |"

    def test_首行不等于表头时不误删(self):
        cells = [["营业收入", "1,234"], ["净利润", "234"]]
        out = linearize_table(cells, header=["项目", "金额"])
        assert "营业收入" in out, "首行是真数据，不能被当成表头删掉"

    def test_records_表头不重复(self):
        cells = [["项目", "2025年"], ["营业收入", "1,234"]]
        recs = table_to_records(cells, header=["项目", "2025年"])
        assert recs == [{"项目": "营业收入", "2025年": "1,234"}]


class TestRecords:
    def test_转成字典(self):
        cells = [["项目", "金额"], ["营业收入", "1,234"]]
        recs = table_to_records(cells)
        assert recs == [{"项目": "营业收入", "金额": "1,234"}]

    def test_重名表头不丢列(self):
        cells = [["项目", "2025年", "2025年"], ["A", "1", "2"]]
        recs = table_to_records(cells)
        assert recs[0]["2025年"] == "1"
        assert recs[0]["2025年_2"] == "2"

    def test_只有表头无数据(self):
        assert table_to_records([["项目", "金额"]]) == []


class TestDropEmptyRows:
    """PyMuPDF 的 text 策略会夹进大量全空行（平安年报实测占 48%）。"""

    def test_删掉全空行(self):
        cells = [["项目", "金额"], [None, None], ["A", "1"], ["", "  "]]
        out = textnorm.drop_empty_rows(cells)
        assert out == [["项目", "金额"], ["A", "1"]]

    def test_有任一单元格有内容就保留(self):
        cells = [["A", None], [None, "1"]]
        assert textnorm.drop_empty_rows(cells) == [["A", None], [None, "1"]]

    def test_全空表不返回空列表(self):
        """全空表由上层丢弃，这里不能把列表清空成 []，否则上层判空逻辑会误判。"""
        cells = [[None, None]]
        assert textnorm.drop_empty_rows(cells) == cells

    def test_空输入(self):
        assert textnorm.drop_empty_rows([]) == []

    def test_不删有内容的行(self):
        cells = [["—", "–"], ["0", "0.00"]]
        assert len(textnorm.drop_empty_rows(cells)) == 2, "破折号和 0 都是有效内容"


class TestDetectMerged:
    def test_行宽不齐报警(self):
        warns = detect_merged_cells([["a", "b", "c"], ["1", "2"]])
        assert any("合并单元格" in w for w in warns)

    def test_大量None报警(self):
        warns = detect_merged_cells([[None, None, "x"], [None, None, "y"]])
        assert any("合并单元格" in w for w in warns)

    def test_正常表不报警(self):
        assert detect_merged_cells([["a", "b"], ["1", "2"]]) == []


class TestNumberLike:
    def test_识别(self):
        for s in ["1", "1,234", "1,234.56", "(1,234)", "-5.5", "11.1%", "（1,234）"]:
            assert is_number_like(s), s

    def test_不误判(self):
        for s in ["营业收入", "1,234亿元", "—", "", "abc"]:
            assert not is_number_like(s), s


class TestPrintedPage:
    def test_识别纯数字页码(self):
        assert textnorm.detect_printed_page_number("12") == "12"

    def test_识别第X页(self):
        assert textnorm.detect_printed_page_number("第 12 页") == "12"

    def test_长文本不误判(self):
        assert textnorm.detect_printed_page_number("本集团2025年营业收入为1,234亿元") is None


class TestTableProfile:
    """表格质量标记。只标记、不删除 —— 因为纯文字的真表格
    （董事简历、关键审计事项）和图文排版的误检在数字占比上无法区分。"""

    def test_数字表判为numeric(self):
        cells = [["项目", "2025年", "2024年"],
                 ["营业收入", "1,234", "1,111"],
                 ["净利润", "234", "210"]]
        label, ratio = textnorm.table_profile(cells)
        assert label == "numeric"
        # "2025年" 带"年"字，不算数字型单元格，所以是 4/9
        assert ratio > 0.4

    def test_纯文字表判为text但不被删(self):
        cells = [["其他主要任职", ""],
                 ["郭先生为平安健康董事长", ""]]
        label, _ = textnorm.table_profile(cells)
        assert label == "text"

    def test_空表(self):
        assert textnorm.table_profile([]) == ("empty", 0.0)
        assert textnorm.table_profile([[None, ""]]) == ("empty", 0.0)

    def test_阈值边界(self):
        # 5 个单元格里 1 个数字 = 0.2，正好是阈值
        cells = [["1,234", "a", "b", "c", "d"]]
        label, ratio = textnorm.table_profile(cells)
        assert ratio == 0.2 and label == "numeric"

    def test_文字型不产生合并告警以外的噪音(self):
        cells = [["短标签", "值"]]
        assert textnorm.table_quality_warnings(cells, "text") == []

    def test_长文字型触发人工复核提示(self):
        long_line = "本集团2025年实现营业收入人民币一万二千三百四十五亿元，同比增长百分之十一点一，主要原因是"
        cells = [["医疗服务", long_line], ["", long_line + "续"]]
        warns = textnorm.table_quality_warnings(cells, "text")
        assert warns and "人工复核" in warns[0]

    def test_数字型不触发复核提示(self):
        cells = [["营业收入", "1,234"]]
        assert textnorm.table_quality_warnings(cells, "numeric") == []


class TestSuspiciousChars:
    """字体编码缺失检测：平安年报第 129 页实测抽出来的是乱码。"""

    def test_正常中文为0(self):
        assert textnorm.suspicious_char_ratio("本公司2025年营业收入为1,234亿元。") == 0.0

    def test_正常符号不算可疑(self):
        s = "■ 项目符号 ● 圆点 ✓ 对勾 β 系数 — 破折号 ① 带圈数字"
        assert textnorm.suspicious_char_ratio(s) == 0.0

    def test_乱码被判为可疑(self):
        # 真实样本：平安年报第 129 页，字体缺 ToUnicode 表
        assert textnorm.suspicious_char_ratio("䉏ߎЮ ͚ܿ ᓰ") > 0.9

    def test_空串(self):
        assert textnorm.suspicious_char_ratio("") == 0.0
        assert textnorm.suspicious_char_ratio("   ") == 0.0
