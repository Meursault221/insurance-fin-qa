"""章节标题识别与章节路径追踪的单元测试（不需要 PDF）。"""

from src.sections import SectionTracker, classify_block


class TestClassifyHeading:
    def test_第X节是标题(self):
        kind, lv = classify_block("第三节 管理层讨论与分析")
        assert kind == "heading"
        assert lv == 1

    def test_固定板块名是标题(self):
        kind, lv = classify_block("公司简介")
        assert kind == "heading" and lv == 1

    def test_括号中文数字是二级(self):
        kind, lv = classify_block("（一）主要财务数据")
        assert kind == "heading" and lv == 2

    def test_中文数字顿号是二级(self):
        kind, lv = classify_block("一、经营概况")
        assert kind == "heading" and lv == 2

    def test_阿拉伯数字点是三级(self):
        kind, lv = classify_block("1. 营业收入")
        assert kind == "heading" and lv == 3

    def test_小数点嵌套是三级四级(self):
        assert classify_block("1.1 分险种保费收入")[1] == 3
        assert classify_block("1.1.1 详细说明")[1] == 4


class TestNotHeading:
    def test_普通句子不是标题(self):
        kind, lv = classify_block("本集团2025年实现营业收入人民币1,234亿元，同比增长11.1%。")
        assert kind == "paragraph" and lv == 0

    def test_数字行不是标题(self):
        # 财报里到处都是这种行，绝不能当标题
        for s in ["1,234.56", "1,234 1,111 11.1", "— — —"]:
            kind, lv = classify_block(s)
            assert kind == "paragraph", f"{s!r} 被误判成标题"

    def test_以句号结尾的不是标题(self):
        kind, lv = classify_block("一、本公司董事会保证本报告内容真实。")
        assert kind == "paragraph"

    def test_超长文本不是标题(self):
        kind, lv = classify_block("一、" + "很长的内容" * 20)
        assert kind == "paragraph"

    def test_空文本(self):
        assert classify_block("")[0] == "other"
        assert classify_block("   ")[0] == "other"

    # ---- 回归：编号本身不能让数字密度超标 ----
    def test_多级编号的标题不被数字密度误杀(self):
        kind, lv = classify_block("1.1.1 详细说明")
        assert (kind, lv) == ("heading", 4), "编号占了太多字符，不能因此判定成数据行"

    def test_纯数字编号加短标题(self):
        kind, lv = classify_block("1.1 分险种")
        assert kind == "heading" and lv == 3

    def test_剥掉编号后没内容的不算标题(self):
        for s in ["1.23", "(1,234)", "1,234.56", "1.", "（1）"]:
            kind, lv = classify_block(s)
            assert kind == "paragraph", f"{s!r} 被误判成标题"

    def test_中文加一串数字的数据行不算标题(self):
        kind, lv = classify_block("示例营业收入 1,234 1,111 11.1")
        assert kind == "paragraph"

    # ---- 回归：整段被括号包住的是注解，不是标题 ----
    def test_括号注解不算标题(self):
        """这些在真实年报里会污染章节路径（平安年报实测 48 处）。"""
        for s in ["（除特别注明外，金额单位为人民币百万元）",
                  "（左坐标）",
                  "(单位：人民币千元)"]:
            kind, lv = classify_block(s, font_size=10.5, body_font_size=9.0)
            assert kind == "paragraph", f"{s!r} 被误判成标题"

    def test_括号编号标题不受影响(self):
        """（一）xxx 不以 ） 结尾，不能被上一条误伤。"""
        assert classify_block("（一）主要财务数据") == ("heading", 2)


class TestSplitNumberPrefix:
    def test_拆分(self):
        from src.sections import split_number_prefix
        assert split_number_prefix("第三节 管理层讨论与分析") == ("第三节", "管理层讨论与分析")
        assert split_number_prefix("（一）主要财务数据") == ("（一）", "主要财务数据")
        assert split_number_prefix("1.1.1 详细说明") == ("1.1.1", "详细说明")
        assert split_number_prefix("一、经营概况") == ("一、", "经营概况")

    def test_没有编号(self):
        from src.sections import split_number_prefix
        assert split_number_prefix("公司简介") == ("", "公司简介")
        assert split_number_prefix("1,234.56") == ("", "1,234.56")


class TestFontSizeRule:
    def test_字号明显偏大判为标题(self):
        kind, lv = classify_block("经营情况讨论", font_size=20.0, body_font_size=10.0)
        assert kind == "heading"

    def test_字号正常且无编号则是正文(self):
        kind, lv = classify_block("经营情况讨论", font_size=10.2, body_font_size=10.0)
        assert kind == "paragraph"

    def test_加粗且略大判为标题(self):
        kind, lv = classify_block("主要会计数据", font_size=11.0,
                                  body_font_size=10.0, bold=True)
        assert kind == "heading"


class TestSectionTracker:
    def test_路径逐级拼接(self):
        tr = SectionTracker()
        assert tr.update("heading", 1, "第三节 管理层讨论与分析") == "第三节 管理层讨论与分析"
        assert tr.update("heading", 2, "（一）主要财务数据").endswith("（一）主要财务数据")
        assert "第三节 管理层讨论与分析" in tr.current

    def test_同级标题替换而非叠加(self):
        tr = SectionTracker()
        tr.update("heading", 1, "第一节 重要提示")
        tr.update("heading", 1, "第二节 公司简介")
        assert tr.current == "第二节 公司简介"

    def test_回到上级时弹出下级(self):
        tr = SectionTracker()
        tr.update("heading", 1, "第一节 A")
        tr.update("heading", 2, "一、B")
        tr.update("heading", 1, "第二节 C")
        assert tr.current == "第二节 C"
        assert "一、B" not in tr.current

    def test_正文沿用栈顶章节(self):
        tr = SectionTracker()
        tr.update("heading", 1, "第三节 管理层讨论与分析")
        before = tr.current
        after = tr.update("paragraph", 0, "本集团2025年……")
        assert after == before

    def test_没有标题时路径为空(self):
        tr = SectionTracker()
        assert tr.update("paragraph", 0, "开头的一段话") == ""

    def test_重复标题去重(self):
        tr = SectionTracker()
        tr.update("heading", 1, "第三节 X")
        tr.update("heading", 1, "第三节 X")
        assert tr.current == "第三节 X"


class TestSentenceFragment:
    """排版折行会把正文切断，断口正好长得像标题。
    平安年报实测：一个折行片段在标题栈里滞留了 52 个 chunk。"""

    def test_折行片段不算标题(self):
        kind, _ = classify_block(
            "中国平安持续践行全球最佳公司治理实践，已经建立了依托本土优势兼具国际标准的",
            font_size=11.5, body_font_size=9.0)
        assert kind == "paragraph"

    def test_以虚词结尾的不算标题(self):
        for s in ["本公司严格按照监管要求开展各项业务并", "上述安排符合同业惯例及",
                  "该项交易已经董事会审议通过、"]:
            assert classify_block(s, font_size=11.5, body_font_size=9.0)[0] == "paragraph", s

    def test_长且带逗号的不算标题(self):
        kind, _ = classify_block("链接供给（供应方）：整合服务网络，构建体系",
                                 font_size=11.5, body_font_size=9.0)
        assert kind == "paragraph"

    def test_真标题不受影响(self):
        for s, lv in [("第三节 管理层讨论与分析", 1), ("（一）主要财务数据", 2),
                      ("一、经营概况", 2), ("1.1 分险种保费收入", 3)]:
            assert classify_block(s) == ("heading", lv), s

    def test_逗号规则的字数边界(self):
        """阈值 20 字：19 字带逗号仍走标题规则，20 字直接判为句子。"""
        short = "甲" * 9 + "，" + "乙" * 9    # 19 字
        long_ = "甲" * 10 + "，" + "乙" * 9   # 20 字
        assert len(short) == 19 and len(long_) == 20
        assert classify_block(short, font_size=12.0,
                              body_font_size=9.0)[0] == "heading"
        assert classify_block(long_, font_size=12.0,
                              body_font_size=9.0)[0] == "paragraph"

    def test_固定板块名不受逗号规则影响(self):
        for s in ["重要提示", "目录", "释义", "公司简介", "审计报告"]:
            assert classify_block(s) == ("heading", 1), s


class TestQuantityNotHeading:
    """数量值长得像章节号："105.1亿元" 会被 "1.1" 编号正则误吃。"""

    def test_数量值不算标题(self):
        for s in ["105.1亿元", "67.79%收购", "1,234万元", "38.5个百分点"]:
            assert classify_block(s)[0] == "paragraph", s

    def test_真的大写编号仍是标题(self):
        for s in ["20.5保险合同的分组", "30.7保险合同计量", "105.1 保险合同"]:
            assert classify_block(s)[0] == "heading", s

    def test_附注编号仍是标题(self):
        assert classify_block("附注25 政府补助")[0] == "heading"
