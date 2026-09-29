"""生成方向 A 的一页中文结论 PDF（作业第四项的收尾件）。

这个脚本**只排版，不跑检索、不调模型**：内容全部是写死在下面的常量，
是 `docs/qa/qa_eval.md` 里 10 道题人工核对之后的结论。所以它可以反复跑、
不花钱、也不会改变任何评测结果。

排版用 PyMuPDF —— 它已经是本项目的既有依赖（抽取 PDF 用的就是它），
所以**不新增任何第三方包**。中文字体用它内置的 `china-s`
（Droid Sans Fallback），不需要外部字体文件，Windows / Linux 上都能跑。
文字是真正的文本层（不是轮廓），可以复制、可以检索。

用法：
    python scripts/make_summary_pdf.py            # 写 docs/qa/summary_onepage.pdf
    python scripts/make_summary_pdf.py -o out.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config as cfg                        # noqa: E402
from src import pymupdf_compat                        # noqa: E402

OUT_DEFAULT = cfg.PROJECT_ROOT / "docs" / "qa" / "summary_onepage.pdf"

# ---------------------------------------------------------------- 内容
# 改结论就改这里。块类型：title 标题 / h 小标题 / p 段落 / li 条目 / table 两列表 / note 小字注
BLOCKS: list[tuple] = [
    ("title", "上市保险公司年报问答知识库——方向 A 实验总结"),

    ("h", "一、实验目的"),
    ("p", "构建一个面向 10 家上市保险公司 2025 年年报的问答知识库：用户以自然语言提问，"
          "系统从年报原文中检索证据，由大模型严格依据检索到的材料作答，并给出可追溯到"
          "公司、报告、页码的引用；材料不足时必须明说不足，不得编造数字或页码。"),

    ("h", "二、数据与方法"),
    ("li", "数据：10 家 A/H 股上市保险公司 2025 年年报原文，共 7,507 个切块。其中 6 家为"
           "港股繁体年报，入库前做简繁归一（只作用于分词与向量层，切块原文不改，"
           "以保证引用逐字可核）。"),
    ("li", "切块：按章节切分并尽量保留表格的行列结构；每个切块携带公司、报告名、页码、"
           "章节、块类型等元数据。"),
    ("li", "检索：BM25（自实现 Okapi，k1=1.2、b=0.1）与向量检索"
           "（BAAI/bge-small-zh-v1.5，512 维）双路召回，用 RRF（k=60）融合，默认取 top-5。"),
    ("li", "生成：将 top-k 切块连同公司/报告/页码元数据送入 DeepSeek（deepseek-chat）。"
           "引用块由程序从切块元数据渲染、不经模型生成；另设相似度阈值、年份一致性、"
           "提示词三层闸门，并核查模型正文里出现的页码是否为材料所有。"),

    ("h", "三、实验结果"),
    ("table", [
        ("测试题数", "10"),
        ("人工判定答对", "8（答案正确率 80%）"),
        ("模型主动拒答（材料不足）", "2"),
        ("gold 页面召回", "9 / 10"),
        ("仅要求主 gold 页时的召回", "5 / 10"),
        ("编造数字 / 页码", "0"),
    ]),
    ("note", "判分口径：公司 + 页码同时命中才算召回 gold；“主 gold 页”指标注中的首要依据页。"
             "答对数由人工对着年报核对得出，脚本不代判。"),

    ("h", "四、典型错误"),
    ("li", "q05「平安去年一年赚了多少钱？」——口语化提问使检索偏向“经营亮点”“内含价值分析”"
           "等页面，未召回净利润所在的财务摘要页（p14/p15）。模型如实回答“材料不足”，未编造，"
           "属检索失败而非生成失败。"),
    ("li", "q09「国寿的保费规模有多大？」——提问用“国寿”“保费规模”，年报原文用“总保费”，"
           "简称与术语不一致，含精确值 7,298.87 亿元的 p13 未被召回。模型只拿到 p12 的"
           "“突破7,000亿元”，明确说明材料未给出确切数值；它也没有把同一指标在 q07 中"
           "已答对的数字挪来补答。"),
    ("li", "另有轻微偏差一处：q10 中模型把材料的“197,191 百万元”自行换算为“1,971.91 亿元”，"
           "与提示词“照抄单位、不要换算”不符。数值本身正确且出处可查，属指令遵循偏差，不是编造。"),

    ("h", "五、结论"),
    ("p", "当前系统的主要瓶颈在检索层，而不是生成层。10 道题中生成层未出现任何编造数字或页码；"
          "两次失败都是“该召回的证据没有进来”，而模型在证据不足时都选择了诚实拒答。"
          "口语化提问、公司简称与财报原文术语之间的映射，是当前最薄弱的环节。"),

    ("h", "六、改进方向"),
    ("li", "1. 查询改写：把口语问题改写成财报术语（如“赚了多少钱”→“归属于母公司股东的净利润”）。"),
    ("li", "2. 同义词与公司简称扩展：“国寿”→“中国人寿”，“保费规模”→“总保费”。"),
    ("li", "3. 提高 top-k：q09 在 top-10 下即可召回，代价是上下文变长、调用成本上升。"),
    ("li", "4. 加强章节与页码信息：用年报自带的目录页定位章节（港股 6 家的章节字段目前不可靠）。"),
    ("li", "5. 对 BM25 与向量检索的融合权重进一步调优，并针对表格型切块调整长度归一化。"),
]

# 从大到小试，取第一个能塞进一页的字号 —— 这样"控制在一页"是算出来的，不靠手调
BODY_SIZES = (11.0, 10.6, 10.2, 9.8, 9.4, 9.0, 8.6, 8.2, 7.8, 7.2, 6.8)

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.,%/\-:]*|\s+|.")


def wrap(text: str, font, size: float, width: float) -> list[str]:
    """按宽度折行。中日韩字符逐字断行，连续的西文/数字（如 7,298.87）保持不拆。"""
    lines: list[str] = []
    cur = ""
    for tok in _TOKEN.findall(text):
        if tok.isspace():
            if cur:
                cur += " "
            continue
        if cur and font.text_length(cur + tok, size) > width:
            lines.append(cur.rstrip())
            cur = tok
        else:
            cur += tok
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


def render(blocks, fitz, font, page, *, page_w, page_h, margin_x, margin_top, margin_bottom,
           size, dry_run: bool) -> float:
    """排一遍。`dry_run=True` 时只算高度不落笔，用来试字号。返回结束时的 y。

    `fitz` 是 `pymupdf_compat.require()` 的返回值，由调用方传进来 —— 模块顶层不 import，
    这样没装 PyMuPDF 时仍然只在使用处报错，不会连 `--help` 都跑不起来。
    """
    width = page_w - 2 * margin_x
    y = margin_top

    def draw_line(s, x, yy, sz, mode=0, color=(0, 0, 0)):
        if not dry_run:
            page.insert_text((x, yy), s, fontname="cn", fontsize=sz,
                             render_mode=mode, color=color)

    def draw_rule(yy, x0, x1, gray=0.75, w=0.5):
        if not dry_run:
            page.draw_line((x0, yy), (x1, yy), color=(gray, gray, gray), width=w)

    for kind, payload in blocks:
        if kind == "title":
            sz = size * 1.62
            if not dry_run:
                draw_rule(y + sz * 0.72, margin_x, page_w - margin_x, gray=0.35, w=1.1)
            y += sz * 0.75
            draw_line(payload, margin_x, y, sz, mode=2)
            y += sz * 0.42
            if not dry_run:
                draw_rule(y, margin_x, page_w - margin_x, gray=0.35, w=1.1)
            y += size * 1.15

        elif kind == "h":
            # 标题前的间距必须**大于一个行高** —— 上一段的基线到标题基线若小于字号，
            # 文字就会叠在一起（之前 0.62 就是这样，字一放大立刻糊成一团）。
            sz = size * 1.14
            y += size * 1.45
            draw_line(payload, margin_x, y, sz, mode=2)
            y += size * 0.62

        elif kind == "p":
            for ln in wrap(payload, font, size, width):
                y += size * 1.5
                draw_line(ln, margin_x, y, size)
            y += size * 0.22

        elif kind == "li":
            indent = size * 1.15
            lines = wrap(payload, font, size, width - indent)
            if not dry_run:
                page.insert_text((margin_x, y + size * 1.5), "·",
                                 fontname="cn", fontsize=size)
            for i, ln in enumerate(lines):
                y += size * 1.5
                draw_line(ln, margin_x + indent, y, size)
            y += size * 0.22

        elif kind == "table":
            rh = size * 1.9
            for i, (k, v) in enumerate(payload):
                if not dry_run and i % 2 == 0:
                    page.draw_rect(fitz.Rect(margin_x - 3, y + 1,
                                             page_w - margin_x + 3, y + rh - 1),
                                   color=None, fill=(0.94, 0.94, 0.94))
                y += rh * 0.72
                draw_line(k, margin_x + 4, y, size)
                draw_line(v, margin_x + width * 0.62, y, size)
                y += rh * 0.28
            y += size * 0.3

        elif kind == "note":
            sz = size * 0.92
            for ln in wrap(payload, font, sz, width):
                y += sz * 1.45
                draw_line(ln, margin_x, y, sz, color=(0.35, 0.35, 0.35))
            y += size * 0.2

    return y


def build(out: Path) -> tuple[Path, float, int]:
    fitz = pymupdf_compat.require()
    font = fitz.Font("china-s")
    page_w, page_h = fitz.paper_size("a4")
    margin_x, margin_top, margin_bottom = 46.0, 42.0, 38.0

    # 先空排一遍量高度。用一个临时 page 承载 dry-run（不落笔也就不会留痕）
    probe = fitz.open()
    p0 = probe.new_page(width=page_w, height=page_h)
    p0.insert_font(fontname="cn", fontbuffer=font.buffer)

    chosen, used = BODY_SIZES[-1], None
    for size in BODY_SIZES:
        end_y = render(BLOCKS, fitz, font, p0, page_w=page_w, page_h=page_h,
                       margin_x=margin_x, margin_top=margin_top,
                       margin_bottom=margin_bottom, size=size, dry_run=True)
        if end_y <= page_h - margin_bottom:
            chosen, used = size, end_y
            break
    probe.close()

    doc = fitz.open()
    page = doc.new_page(width=page_w, height=page_h)
    page.insert_font(fontname="cn", fontbuffer=font.buffer)
    end_y = render(BLOCKS, fitz, font, page, page_w=page_w, page_h=page_h,
                   margin_x=margin_x, margin_top=margin_top,
                   margin_bottom=margin_bottom, size=chosen, dry_run=False)

    # 内嵌的 Droid Sans Fallback 是整个字库，不裁剪的话一页纸也要 1.7 MB。
    # subset_fonts() 只保留用到的字，能压到 55 KB 左右；它需要 fontTools，
    # 没装就跳过 —— 文件大一点而已，不该因此生成失败。
    try:
        doc.subset_fonts()
    except Exception as e:                                    # noqa: BLE001
        print(f"提示：字体子集化跳过（{type(e).__name__}），文件会大一些")

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out), deflate=True, garbage=4)
    doc.close()
    return out, chosen, used if used is not None else end_y


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成方向 A 的一页结论 PDF")
    ap.add_argument("-o", "--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args(argv)

    out, size, end_y = build(args.out)
    fitz = pymupdf_compat.require()
    d = fitz.open(str(out))
    n = d.page_count
    text = d[0].get_text()
    d.close()

    print(f"已生成：{out}")
    print(f"页数：{n}（要求 1）")
    print(f"正文字号：{size}pt   排版末端 y={end_y:.0f} / 可用 {fitz.paper_size('a4')[1]:.0f}")
    print(f"文本层字数：{len(text.strip())}（能复制、能检索，不是图片）")
    return 0 if n == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
