# web —— 问答页面（第三阶段）

`app.py` 就是页面本体，Streamlit 单文件。

## 启动

```bash
# 项目根目录下
streamlit run web/app.py
```

浏览器打开 <http://localhost:8501>（Streamlit 会自动开一个标签页）。

不配大模型也能开 —— 页面会正常显示检索结果和引用，只是不生成答案正文。
要生成答案就在项目根目录建 `.env`（模板见根目录 `.env.example`），填一个 key：

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

或者任意 OpenAI 兼容接口：

```
OPENAI_API_KEY=sk-xxxxxxxx
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

本地 Ollama 也可以（不联网、不花钱）：

```
OPENAI_API_KEY=ollama
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL=qwen2.5:7b
```

## 页面要素

- **输入框**：问一个问题。下面有 5 个示例按钮，点一下直接填进去
- **侧栏**：检索方式（混合/BM25/向量）、送给模型的片段数、限定公司、
  当前大模型配置、相似度拒答阈值
- **答案区**：模型生成的答案正文。拒答时显示拒答原因和怎么改问法
- **引用区（作业硬性要求）**：每条来源的引用块直接取自 chunk 元数据
  （**不是模型写的**）
  ```
  公司：中国平安
  报告：2025年年度报告
  页码：364页
  章节：财务报表
  ```
  被答案正文引用到的那些会标一个 ✅
- **原文片段**：默认展开，可以核对引用说的是不是真话
- **每个片段带**：融合得分、检索方式、chunk 类型、字数、股票代码、chunk_id

**注意：页面本身没有任何检索或提示词逻辑**，全部来自 `src/qa.py` 和 `src/retriever.py`。
改检索要改那边，改页面表现只改 `app.py`。

## 自测页面（不用开浏览器）

```bash
python -c "
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('web/app.py', default_timeout=300)
at.run()
at.text_input(key='question').set_value('中国平安2025年归属于母公司股东的净利润是多少？')
at.run()
print('异常:', list(at.exception))
print('引用:', [c.value for c in at.code])
"
```

## 注意

`streamlit run` 会打开浏览器。作业要求交**问答页面截图**，
截图请放在 `web/static/screenshots/`（该目录已被 .gitignore 排除 PNG，
如果截图要提交，请把 .gitignore 里那条注释掉）。
