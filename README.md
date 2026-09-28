# 基于向量检索与 BM25 的上市保险公司财报问答知识库

> 课程作业 · 方向 A
> **当前进度：第一阶段（项目基础架构）已完成。** 检索与问答页面尚未实现。

从 10 家上市保险公司的年报 PDF 里抽取带来源信息的文本与表格，
建立「向量检索 + BM25」混合检索，做一个能提问、能回答、**每个答案都带出处**的问答页面。

---

## 一、当前完成到哪一步

| 阶段 | 内容 | 状态 |
|---|---|---|
| **一** | 项目结构、PDF 解析、表格还原、切分、来源信息 | ✅ **已完成** |
| 二 | 向量检索 + BM25 混合检索、索引落盘 | ⬜ 未开始 |
| 三 | 问答页面、来源展示 | ⬜ 未开始 |
| 四 | 10 道测试题 + 记录表、一页总结 PDF | ⬜ 未开始 |

**这一阶段只做了"能读、能切、能溯源"，没有下载任何真实财报，也没有实现检索。**
仓库里唯一的 PDF 是脚本生成的**合成测试样本**，不含任何真实财务数据。

---

## 二、目录结构

```
insurance-fin-qa/
├── README.md                     ← 本文件
├── requirements.txt              第一阶段依赖（纯 wheel，Windows 直接装）
├── requirements-retrieval.txt    第二阶段依赖（现在不用装）
├── pytest.ini / conftest.py      测试配置
│
├── configs/
│   └── companies.yaml            ⭐ 10 家公司的元数据（股票代码、报告名、来源 URL）
│
├── data/
│   ├── raw/                      ⭐ 你要把下载的年报 PDF 放这里
│   │   └── README.md             命名规范说明
│   ├── extracted/                抽取产物：每份 PDF 一个结构化 JSON
│   ├── chunks/                   切分产物：每份报告一个 JSONL
│   └── samples/                  小样本 PDF（调试用）
│
├── index/
│   ├── bm25/                     第二阶段：BM25 索引
│   └── vector/                   第二阶段：向量索引
│
├── src/                          ⭐ 核心代码
│   ├── schema.py                 数据结构（DocumentMeta / TextBlock / TableBlock / Chunk）
│   ├── config.py                 路径 + 公司配置加载
│   ├── textnorm.py               文本清洗 + 表格线性化
│   ├── sections.py               章节标题识别 + 章节路径追踪
│   ├── tables.py                 表格抽取后端（PyMuPDF / pdfplumber）
│   ├── pdf_extractor.py          ⭐ PDF → 结构化文档
│   ├── chunker.py                ⭐ 结构化文档 → 检索单元
│   ├── pipeline.py               单份 / 批量编排
│   ├── pymupdf_compat.py         PyMuPDF 导入兼容
│   └── cli.py                    命令行入口
│
├── scripts/
│   ├── make_demo_pdf.py          生成合成测试 PDF（无需下载任何东西就能跑通）
│   └── run_single_pdf.py         单份 PDF 测试脚本（可直接改常量）
│
├── tests/                        117 个单元测试，全部通过
│   ├── fixtures/                 测试样本（合成 PDF）
│   └── test_*.py
│
├── docs/
│   ├── phase1_architecture.md    第一阶段设计说明（含数据流、字段表）
│   └── data_download_guide.md    年报下载指引（交易所网站入口）
│
└── web/                          第三阶段：问答页面
```

---

## 三、安装

Windows 11 + Python 3.10~3.12，全部是纯 wheel 包，**不需要 Java，不需要 Ghostscript**。

```bash
cd insurance-fin-qa

# 建议先建虚拟环境
python -m venv .venv
.venv\Scripts\activate          # PowerShell/CMD
# source .venv/Scripts/activate # Git Bash

pip install -r requirements.txt

# reportlab 只用来生成合成测试样本，不是运行依赖
pip install reportlab

# 检查环境
python -m src.cli doctor
```

`doctor` 会检查 PyMuPDF、两个表格后端、配置文件、目录是否就绪。

---

## 四、怎么跑（现在还不需要任何真实 PDF）

### 步骤 1：生成一份合成测试 PDF

```bash
python scripts/make_demo_pdf.py
```

生成 `tests/fixtures/demo_company_2025_annual.pdf` —— 两页，含两个带框线的表格、多级章节标题、页脚页码。
**这份文件是脚本编的占位内容，不是任何公司的财报。**

### 步骤 2：解析它

```bash
python -m src.cli extract tests/fixtures/demo_company_2025_annual.pdf \
    --company "示例公司" --key demo --year 2025 \
    --report-name "2025年年度报告（合成样本）" \
    --preview 5
```

输出：

```
文档      : 示例公司 / 2025年年度报告（合成样本）
页数      : 2
文本块    : 26
表格      : 2（32 个单元格）
正文数量  : 352 字符
解析器    : PyMuPDF 1.28.2 / 表格后端=pymupdf
切分结果  : 7 个 chunk，覆盖 2 页
结构化 JSON: data/extracted/demo_2025_annual.json
chunks     : data/chunks/demo_2025_annual.jsonl
```

`--preview 5` 会打印前 5 个 chunk 及其来源信息，格式就是作业要求的那种：

```
--- demo_2025_annual_p0001_c0003  [table] 151 字 ---
公司：示例公司
报告：2025年年度报告（合成样本）
页码：第 1 页
章节：第二节 主要财务数据
------------------------------------------------------------------
〔表格〕单位：人民币百万元（以下为占位数字）
| 项目 | 2025年 | 2024年 | 变动(%) |
| --- | --- | --- | --- |
| 示例营业收入 | 1,234 | 1,111 | 11.1 |
...
```

### 步骤 3：换成真实 PDF

```bash
python -m src.cli extract data/raw/pingan_2025_annual.pdf
```

文件名能对上 `configs/companies.yaml` 里的约定时，公司名、报告名、股票代码会自动填好，不用敲一堆参数。

### 步骤 4：跑测试

```bash
python -m pytest
```

---

## 五、常用命令

| 命令 | 作用 |
|---|---|
| `python -m src.cli doctor` | 检查依赖与目录 |
| `python -m src.cli companies` | 列出 10 家公司配置 |
| `python -m src.cli extract <pdf>` | 处理单份 PDF |
| `python -m src.cli extract <pdf> --no-write --preview 3` | 只跑不落盘，预览 |
| `python -m src.cli extract <pdf> --table-backend pdfplumber` | 换表格识别后端 |
| `python -m src.cli batch` | 批量处理 `data/raw/` 下所有 PDF |
| `python -m src.cli preview data/chunks/xxx.jsonl` | 查看已生成的 chunks |

常用参数：

| 参数 | 说明 |
|---|---|
| `--profile <key>` | 指定公司（见 `companies` 命令输出） |
| `--company / --key / --year / --report-name` | 手动指定来源信息 |
| `--table-backend pymupdf\|pdfplumber\|none` | 表格识别后端 |
| `--max-chars / --min-chars / --overlap` | 切分参数（默认 700/120/100） |
| `--include-table-text` | 把表格区域的原始文本行也做成 chunk（默认不，避免数字重复） |
| `--no-tables` | 完全跳过表格 |

---

## 六、设计上的几个关键决定

### 1. 数字只搬运，不加工
不做千分位清洗、不做单位换算、不把 `—` 或空单元格补成 `0`。
PDF 里是 `1,234,567.89`，抽出来就是 `1,234,567.89`。
任何"美化"都会让财报数据失真，而这正是最不能出错的地方。

### 2. 表格双形态保存
每张表同时存 `cells`（二维数组，真正的行列结构）和 `text`（markdown 管道表）。
- `cells` → 第二阶段精确取数（"中国平安 2025 年归母净利润是多少"）
- `text` → 喂给向量模型和 BM25

### 3. 表格是原子 chunk，绝不切开
财报表格从中间劈开等于作废 —— 数字和科目对不上号。

### 4. chunk 不跨页
每个 chunk 只归属一页，`page` 字段永远精确。
代价是段落可能被页边界截断，换来的是**引用页码绝对可信**。

### 5. 表格区域的文本不丢，但默认不重复收录
PyMuPDF 抽表格时，表格里的文字同时也会以普通文本块形式出现。
抽取层把它们标记成 `kind="table_text"` **完整保留在 JSON 里**，
由 chunker 决定要不要收录（默认不），避免同一批数字在索引里出现两遍。

### 6. 认不出公司名就报错，绝不瞎猜
来源标错比没有来源更糟。文件名对不上配置时，脚本会拒绝运行并要求显式指定。

---

## 七、下一步：你需要做什么

### 立刻要做的

1. **核对 `configs/companies.yaml` 里的股票代码**（现在都标着 `verified: false`）。
   尤其港股代码，请到交易所页面确认后改成 `true`。

2. **下载 10 份 2025 年年度报告 PDF**，放到 `data/raw/`，按约定命名：
   ```
   pingan_2025_annual.pdf
   china_life_2025_annual.pdf
   cpic_2025_annual.pdf
   picc_2025_annual.pdf
   new_china_life_2025_annual.pdf
   china_taiping_2025_annual.pdf
   china_re_2025_annual.pdf
   sunshine_2025_annual.pdf
   zhongan_2025_annual.pdf
   picc_pc_2025_annual.pdf
   ```
   下载入口见 `docs/data_download_guide.md`。
   **下载后把真实 URL 填进 yaml 的 `source_url` 字段** —— 最终报告的来源可追溯性靠它。

3. **先只放 1 份，跑通再说**：
   ```bash
   python -m src.cli extract data/raw/pingan_2025_annual.pdf --preview 5
   ```
   然后**人工抽查**：
   - 打开 `data/extracted/pingan_2025_annual.json`，随便搜一个你记得的数字，
     确认 `page` 字段和 PDF 阅读器里显示的页码一致
   - 看 `tables[].cells` 的行列有没有错位
   - 看 `blocks[].section` 的章节路径对不对

4. 单份没问题了再批量：
   ```bash
   python -m src.cli batch
   ```

### 需要你确认后再做的

- 第二阶段（混合检索）：需要装 `sentence-transformers` + `faiss`，会拉约 2GB 的 torch。
  国内网络建议先配好镜像源再装。
- 问答页面用 Streamlit 还是 FastAPI + 前端，你可以决定。

---

## 八、已知限制

- **扫描件不支持**：纯图片版 PDF 抽不出文字，`doctor` 和抽取结果里会告警
  （`scanned_pages` 字段）。这类报告需要先做 OCR。
- **复杂合并单元格**：跨行跨列的复杂表头可能错位，`warnings` 字段会提示
  （"可能存在合并单元格"）。抽出后需要人工核对。
- **印刷页码 vs 物理页号**：很多年报正文的页脚印刷页码和 PDF 物理页号不一致。
  两个都存了（`page` 是物理页号，`printed_page` 是页脚数字），
  引用时以 `page` 为准（可复现），需要时一并显示。
- **页面字体编码缺失**：部分年报的图表/组织架构图用了缺少 ToUnicode 映射表的
  子集字体，抽出来是乱码（平安年报实测第 19-24、119、129 页）。**这些页面
  正文段落是好的，坏的是图形里的标签文字**，需要 OCR 才能救回。抽取结果里通过
  `stats.garbled_pages` 和页级 `warnings` 标出，不会静默污染索引。

- **表格识别有约三分之一是"文字型"**：中文年报大量使用三线表，PyMuPDF 的
  `text` 策略靠文字对齐推断列边界，会把一些图文排版区域（董事长致辞的两栏正文、
  信息图）也认成表格。实测平安年报 182 张表里 63 张数字占比 < 20%。
  **数据一个都没丢**（都进了 chunk），但检索质量会受影响。每张表都带
  `quality`（`numeric`/`text`）和 `numeric_ratio` 字段，供第二阶段区别对待；
  部分还带"疑似图文排版被误识别"的告警。**没有做自动删除** —— 因为纯文字的
  真表格（董事简历、关键审计事项）和误检在数字占比上无法区分，删了会误杀。

- **少量表格列被合并**：三线表推断列边界时会偶发把多列并成一列
  （如 `(8,200) (7,262) (8,200) (7,262` 挤在一个单元格里）。
  **数字本身没丢**，但二维结构失真，精确取数时需要人工核对这几张表。

- **页眉页脚**：公司名书眉和页码型页脚都会剔除（`_mark_running_headers`
  按"上下边距内 + 去数字后重复出现在 ≥4 页"判定）。实测把章节路径被页眉
  污染的比例从 26.7% 降到 0。极少数短标题若恰好重复出现在很多页的边距里，
  有被误剔的可能。
