# 人工复核留档

这里是抽取质量排查时渲染出来的页面截图，用于佐证 README「已知限制」里的结论。
不是程序产物，不会自动生成。

| 文件 | 页面 | 说明 |
| --- | --- | --- |
| `check_p20.png` | 物理第 20 页（印刷第 16 页） | 信息图页。正文段落提取完好（"约 9,000 万""7 个百分点""2 倍"这些数字正文里都有），但图内标签（交易 / 融资 / 理赔 / 救急）在抽取结果里是乱码 |
| `check_p129.png` | 物理第 129 页（印刷第 125 页） | 组织机构图页。图中部门名称肉眼可读，但文字层用了缺少 ToUnicode 映射表的子集字体，抽出来是乱码；同页正文段落正常 |

复现方式：

```bash
python -c "
from src import pymupdf_compat
fitz = pymupdf_compat.require()
doc = fitz.open('data/raw/pingan_2025_annual.pdf')
doc[19].get_pixmap(dpi=110).save('docs/qa/check_p20.png')
"
```

结论：这两类页面需要 OCR 才能救回图形内的文字。抽取结果里通过
`stats.garbled_pages` 和页级 `warnings` 标出，不会静默污染索引。
