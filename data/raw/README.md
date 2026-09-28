# data/raw —— 原始年报 PDF 放这里

这个目录**不进 git**（文件大且有版权）。目录本身保留。

## 命名规范

脚本按文件名自动识别公司和年份，命名对上 `configs/companies.yaml` 就不用敲一堆参数。

```
pingan_2025_annual.pdf           中国平安
china_life_2025_annual.pdf       中国人寿
cpic_2025_annual.pdf             中国太保
picc_2025_annual.pdf             中国人保
new_china_life_2025_annual.pdf   新华保险
china_taiping_2025_annual.pdf    中国太平
china_re_2025_annual.pdf         中国再保险
sunshine_2025_annual.pdf         阳光保险
zhongan_2025_annual.pdf          众安在线
picc_pc_2025_annual.pdf          中国财险
```

命名规则：`<公司key>_<年份>_<annual|semi>.pdf`

- 年份抓 4 位数字，写 2024 就是 2024 年
- `semi` / `half` / `半年` / `中期` 会被识别成半年报
- 认不出来也没关系，用 `--company` 显式指定即可
- **脚本不会瞎猜公司名**：对不上就报错要你指定，因为来源标错比没有来源更糟

## 下载完之后

1. 把真实下载地址填进 `configs/companies.yaml` 对应公司的 `source_url` 字段。
   不要编造 URL —— 留空也比编一个强。
2. 核对股票代码，把 `verified` 改成 `true`。
3. 先拿 1 份跑通：
   ```bash
   python -m src.cli extract data/raw/pingan_2025_annual.pdf --preview 5
   ```
4. 人工抽查页码和表格行列没错位，再批量：
   ```bash
   python -m src.cli batch
   ```

## 注意

- 优先用**交易所官网下载的原始 PDF**，不要用第三方转格式过的版本
- 如果 PDF 是扫描件（整页图片），抽不出文字。抽取结果里会告警并列出
  `scanned_pages`，这类需要先做 OCR
- 港股报告是繁体，A 股是简体，两者都会被正常处理
