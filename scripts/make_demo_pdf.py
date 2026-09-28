"""生成一份**合成的**测试 PDF，用来在没有任何真实财报的情况下先把流程跑通。

⚠️ 重要声明：
    这份 PDF 里的所有文字和数字都是脚本临时编的占位符，
    **不是任何一家公司的财务数据**，也不代表任何真实财报的格式规范。
    它唯一的用途是：验证「抽取 -> 表格还原 -> 章节识别 -> 页码归属 -> 切分」
    这条链路能不能跑通，以及让单元测试有一个稳定的输入。

真实数据请从交易所网站下载，放到 data/raw/。

用法:
    python scripts/make_demo_pdf.py
    # 默认输出 tests/fixtures/demo_company_2025_annual.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src import config  # noqa: E402

DEFAULT_OUT = config.FIXTURES_DIR / "demo_company_2025_annual.pdf"


def build(out_path: Path) -> Path:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                        Spacer, Table, TableStyle)
    except ImportError:
        raise SystemExit("需要 reportlab：pip install reportlab\n"
                         "（它只用来生成测试样本，不是项目运行依赖）")

    # reportlab 自带的中文 CID 字体，无需外部字体文件
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    FONT = "STSong-Light"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    h1 = ParagraphStyle("h1", fontName=FONT, fontSize=18, leading=24, spaceAfter=6)
    h2 = ParagraphStyle("h2", fontName=FONT, fontSize=14, leading=20, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("body", fontName=FONT, fontSize=10.5, leading=17, spaceAfter=6)
    note = ParagraphStyle("note", fontName=FONT, fontSize=8.5, leading=12,
                          textColor=colors.red)

    def footer(canvas, doc_):
        """画页脚页码 —— 顺便测试我们的印刷页码识别。"""
        canvas.saveState()
        canvas.setFont(FONT, 9)
        canvas.drawCentredString(A4[0] / 2, 15 * mm, str(doc_.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=20 * mm, bottomMargin=25 * mm,
        title="示例公司 2025 年年度报告（合成测试样本）",
    )

    story = []
    story.append(Paragraph("示例公司 2025 年年度报告", h1))
    story.append(Paragraph(
        "【本文件为软件测试用合成样本 · 所有数字均为占位符 · 不含任何真实财务数据】", note))
    story.append(Spacer(1, 8))
    story.append(Paragraph("第一节 重要提示", h2))
    story.append(Paragraph(
        "本段落用于测试正文抽取与断句。示例公司是一家虚构的测试主体，"
        "本节内容仅用于验证 PDF 文本抽取管线是否正常工作。"
        "下面出现的一切数字都只是占位符，不对应任何真实报表科目。", body))

    story.append(Paragraph("第二节 主要财务数据", h2))
    story.append(Paragraph("单位：人民币百万元（以下为占位数字）", body))

    data1 = [
        ["项目", "2025年", "2024年", "变动(%)"],
        ["示例营业收入", "1,234", "1,111", "11.1"],
        ["示例净利润", "234", "210", "11.4"],
        ["示例总资产", "12,345,678", "11,000,000", "12.2"],
        ["示例每股收益(元)", "1.23", "1.10", "11.8"],
    ]
    t1 = Table(data1, colWidths=[52 * mm, 32 * mm, 32 * mm, 26 * mm])
    t1.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    story.append(t1)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "注：本表为合成占位数据。真实财报中此处通常是合并资产负债表的主要科目。", body))

    story.append(PageBreak())

    story.append(Paragraph("第三节 经营情况讨论与分析", h2))
    story.append(Paragraph("（一）示例业务分部情况", h2))
    story.append(Paragraph(
        "本页用于测试第二页的页码归属是否正确。"
        "每一条抽取出来的文本都应该带上它真实的物理页序号。", body))

    data2 = [
        ["示例分部", "示例保费收入", "示例赔付支出"],
        ["分部甲", "8,888", "5,555"],
        ["分部乙", "4,444", "2,222"],
        ["合计", "13,332", "7,777"],
    ]
    t2 = Table(data2, colWidths=[46 * mm, 48 * mm, 48 * mm])
    t2.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    story.append(t2)

    story.append(Paragraph("三、示例风险提示", h2))
    story.append(Paragraph(
        "这一节存在的意义是让章节路径至少有两级，"
        "好验证「第三节 经营情况讨论与分析 > 三、示例风险提示」这种路径拼接。", body))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out_path


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    path = build(out)
    print(f"[OK] 已生成合成测试 PDF: {path}")
    print()
    print("下一步运行:")
    print(f'  python -m src.cli extract "{path}" --company "示例公司" --year 2025 '
          f'--report-name "2025年年度报告（合成样本）" --preview 3')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
