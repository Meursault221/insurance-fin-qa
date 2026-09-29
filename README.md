# 基于向量检索与 BM25 的上市保险公司财报问答知识库

> 课程作业 · 方向 A
> **当前进度：第三阶段（问答页面 + 大模型回答 + 页码引用）已完成，闭环跑通。**
> 下一步是作业要求的 10 道题人工测试与记录。

从 10 家上市保险公司的年报 PDF 里抽取带来源信息的文本与表格，
建立「向量检索 + BM25」混合检索，做一个能提问、能回答、**每个答案都带出处**的问答页面。

---

## 一、当前完成到哪一步

| 阶段 | 内容 | 状态 |
|---|---|---|
| **一** | 项目结构、PDF 解析、表格还原、切分、来源信息 | ✅ **已完成** |
| **二** | 向量检索 + BM25 混合检索、索引落盘、检索层评测 | ✅ **已完成** |
| **三** | 问答页面、检索 → 大模型 → 带页码引用的答案 | ✅ **已完成** |
| 四 | 10 道测试题 + 记录表、一页总结 PDF | 🟡 检索层 29 题已完成，问答层 10 题待人工做 |

第二阶段实测（29 道题覆盖 10 家公司，top-10）：

| 检索方式 | 题目命中率 | 目标召回率 | MRR | Top-1 命中 |
|---|---|---|---|---|
| BM25 | 24/29 (82.8%) | 83.9% | 0.535 | 11/29 |
| 向量 | 19/29 (65.5%) | 64.5% | 0.273 | 3/29 |
| **混合** | **26/29 (89.7%)** | **90.3%** | **0.633** | **14/29** |
| 公司识别 | **29/29 = 100%** | | | |

设计说明、融合配置对比、已知局限见 [`docs/phase2_retrieval.md`](docs/phase2_retrieval.md)；
逐题明细见 [`docs/qa/retrieval_eval.md`](docs/qa/retrieval_eval.md)。

第三阶段在这个检索层之上接了问答大模型：问题 → 混合检索 → 命中的 chunk
（正文 + 公司/报告/页码/章节）交给模型 → 模型写答案 → **引用块由程序从 chunk
元数据渲染**。所以页面上显示的公司、报告、页码永远不会是模型编的。详见
[`docs/phase3_qa.md`](docs/phase3_qa.md)。

---

## 二、目录结构

```
insurance-fin-qa/
├── README.md                     ← 本文件
├── requirements.txt              第一阶段依赖（纯 wheel，Windows 直接装）
├── requirements-retrieval.txt    第二阶段依赖（含 Windows 上 torch CPU 版装法）
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
│   └── samples/                  产出物：抓取台账 / 批量校验报告 / 统一来源清单
│
├── index/
│   ├── bm25/                     第二阶段：BM25 倒排索引（62,954 词条）
│   └── vector/                   第二阶段：向量索引（7507×512 float32）
│
├── src/                          ⭐ 核心代码
│   ├── schema.py                 数据结构（DocumentMeta / TextBlock / TableBlock / Chunk）
│   ├── config.py                 路径 + 公司配置加载
│   ├── textnorm.py               文本清洗 + 表格线性化
│   ├── sections.py               章节标题识别 + 章节路径追踪
│   ├── tables.py                 表格抽取后端（PyMuPDF / pdfplumber）
│   ├── pdf_extractor.py          ⭐ PDF → 结构化文档
│   ├── chunker.py                ⭐ 结构化文档 → 检索单元
│   ├── bm25.py                   第二阶段：BM25（打分前按公司布尔过滤）
│   ├── vector_index.py           第二阶段：向量索引（BGE-small-zh）
│   ├── hybrid.py                 第二阶段：RRF / weighted / union 融合
│   ├── tokenize.py               第二阶段：中文分词 + 简繁归一
│   ├── retriever.py              ⭐ 第二阶段：统一检索入口
│   ├── llm.py                    第三阶段：大模型调用 + 防伪造核查（纯标准库）
│   ├── qa.py                     ⭐ 第三阶段：检索 → 三层闸门 → 大模型 → 带引用答案
│   ├── pipeline.py               单份 / 批量编排
│   ├── pymupdf_compat.py         PyMuPDF 导入兼容
│   └── cli.py                    命令行入口（search / ask 子命令）
│
├── scripts/
│   ├── make_demo_pdf.py          生成合成测试 PDF（无需下载任何东西就能跑通）
│   ├── run_single_pdf.py         单份 PDF 测试脚本（可直接改常量）
│   ├── fetch_reports.py          从官方渠道抓年报（巨潮资讯网 / 港交所披露易）
│   ├── record_provenance.py      把下载记录写回 configs/companies.yaml
│   ├── phase1_batch_report.py    Phase 1.5 批量校验，产出文档质量报告
│   ├── build_provenance.py       生成统一来源清单（现算 sha256 并与配置比对）
│   ├── build_index.py            第二阶段：建索引（带 chunks 指纹校验）
│   ├── eval_retrieval.py         第二阶段：检索层评测 + 融合配置对比（--sweep）
│   └── qa_eval.py                第三阶段：问答层测试记录表（作业第四项用）
│
├── tests/                        294 个单元测试，全部通过（229 + 第三阶段 65）
│   ├── fixtures/                 测试样本（合成 PDF）
│   └── test_*.py
│
├── docs/
│   ├── phase1_architecture.md    第一阶段设计说明（含数据流、字段表）
│   ├── phase2_retrieval.md       ⭐ 第二阶段设计说明（含融合对比、已知局限）
│   ├── phase3_qa.md              ⭐ 第三阶段设计说明（三层闸门、防伪造）
│   ├── data_download_guide.md    年报下载指引（交易所网站入口）
│   └── qa/                       检索层评测结果（逐题明细 + 汇总表）
│
└── web/
    ├── app.py                    ⭐ 第三阶段：Streamlit 问答页面
    └── README.md
```

根目录另有 `.env.example`（大模型配置模板，复制成 `.env` 使用）。

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

第二阶段（混合检索）另装一份依赖。**第一行不能省** —— Windows 上 PyPI 默认给的
torch wheel 是 CUDA 版（约 2.4GB），而本项目的向量模型跑在 CPU 上：

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch   # 约 200MB
pip install -r requirements-retrieval.txt

# 建索引（chunks 没变会自动跳过）
python scripts/build_index.py

# 试一句（这是一道评测题的原文，top-1 就是正确来源）
python -m src.cli search "中国平安2025年归属于母公司股东的净利润是多少？" -k 5 --show-text 200

# 跑检索层评测（29 道题，逐路对比）
python scripts/eval_retrieval.py
python scripts/eval_retrieval.py --sweep        # 外加融合配置对比
```

**不需要 faiss，也不需要 rank-bm25** —— 原因见 `requirements-retrieval.txt`。

第三阶段（问答页面）多一个 `streamlit`，已经写在同一个文件里，不用再单独装：

```bash
# 配大模型（可选 —— 不配也能跑，只是不生成答案正文）
copy .env.example .env          # Git Bash: cp .env.example .env
# 编辑 .env，填一个 key（Anthropic 或任意 OpenAI 兼容接口都行）

# 开问答页面
streamlit run web/app.py
```

浏览器打开 <http://localhost:8501>，在输入框里提问即可。页面左侧可以调检索方式、
材料条数、限定公司；右侧答案下面是引用来源，每条都能展开看原文片段。

命令行版本（不开浏览器也能测同一套流程）：

```bash
python -m src.cli ask "中国平安2025年归属于母公司股东的净利润是多少？" -k 5 --show-text 120
```

---

## 四、怎么跑

前两步用一份**脚本生成的合成 PDF** 就能跑通，不需要下载任何东西。

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
页码：1页
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
| `python -m src.cli search "问题"` | 第二阶段：只检索，不生成答案 |
| `python -m src.cli ask "问题"` | 第三阶段：检索 + 大模型 + 带页码引用的答案 |
| `streamlit run web/app.py` | 第三阶段：问答页面 |

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

## 七、数据是怎么来的

10 家公司、2025 年年报，全部从官方渠道下载（巨潮资讯网 / 港交所披露易），
**没有用任何第三方转载的 PDF**。每份文件的下载地址和 sha256 记在
`configs/companies.yaml` 和 `data/samples/final_provenance.json` 里。

```
pingan_2025_annual.pdf            中国平安      601318.SH / 2318.HK
china_life_2025_annual.pdf        中国人寿      601628.SH / 2628.HK
cpic_2025_annual.pdf              中国太保      601601.SH / 2601.HK
picc_2025_annual.pdf              中国人保      601319.SH / 1339.HK
new_china_life_2025_annual.pdf    新华保险      601336.SH / 1336.HK
china_taiping_2025_annual.pdf     中国太平      0966.HK
china_re_2025_annual.pdf          中国再保险    1508.HK
sunshine_2025_annual.pdf          阳光保险      6963.HK
zhongan_2025_annual.pdf           众安在线      6060.HK
picc_pc_2025_annual.pdf           中国财险      2328.HK
```

处理流程：`python -m src.cli extract data/raw/<file>.pdf` → `python -m src.cli batch`，
产物在 `data/extracted/`（结构化 JSON）和 `data/chunks/`（检索单元 JSONL）。
10 份报告合计 **7507 个 chunk**。

### 数据完整性

第三阶段开始时复核过一遍，结论写在 `docs/phase3_qa.md`：
7507 条 chunk 正文与磁盘上的 `data/chunks/*.jsonl` **逐字节一致**（0 条不一致）；
简繁归一（只作用于分词/向量，不改 chunk 正文）前后，
7507 条 chunk 里 **0 条**的数字串发生变化 —— 全文 122,624 个数字串、
443,532 个数字字符原样保留。

### 第三阶段已完成的部分

- 问答页面用 **Streamlit**（`web/app.py`，一个文件，不用写前端）。
- 已接入问答大模型，**不装任何 SDK**（标准库 `urllib` 直接发 HTTP），
  Anthropic 和任意 OpenAI 兼容接口都支持。
- 没配 API key 也能开页面：照常显示检索结果和引用，只是不生成答案正文。

### 还剩下的（作业第四项）

- **10 道题的人工测试与记录**：题目、召回内容、引用、答案已经能自动记录，
  "答案是否正确 / 错在哪里"两列需要人工核对后填。见
  [`docs/qa/README.md`](docs/qa/README.md)。
- 一页总结 PDF。

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
