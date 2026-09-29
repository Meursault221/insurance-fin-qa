"""大模型调用层（第三阶段）。

这一层只干两件事：**把检索到的材料拼成提示词**，以及**把回答取回来**。
它不认识 Retriever，也不碰索引 —— 输入是纯文本，输出是纯文本，
所以可以完全离线做单元测试（见 tests/test_qa.py 里的假客户端）。

## 为什么用标准库发 HTTP，而不是装 SDK

`ANTHROPIC_API_KEY` 和 "OpenAI 兼容" 这两种接口都只是「POST 一段 JSON、
取回一段 JSON」。用 `urllib` 就够了，于是**这一层不需要新增任何依赖** ——
考虑到本项目要求"Windows 下容易装、容易跑"，少一个包少一份麻烦：
DeepSeek / 通义 / 智谱 / Kimi / Ollama / vLLM 全都提供 OpenAI 兼容接口，
换一个 `LLM_BASE_URL` 就能用，不用换代码、也不用再装东西。

需要流式输出、重试、tool use 这些能力时再换官方 SDK 也不迟。

## 防伪造

作业和本项目都硬性要求**不能编造数字和页码**。这里的做法是双保险：

1. 提示词里要求模型只依据给定材料，且每条事实标注材料编号 `[1]` `[2]`。
2. **引用由程序渲染，不由模型生成** —— 页面上展示的"公司 / 报告 / 页码"
   一律取自检索命中的 chunk 元数据（`Chunk.citation()`），模型写什么都不影响。
   即使模型在正文里胡写一个页码，也不会变成引用块。
3. 另外用 `find_suspicious_pages()` 扫一遍模型正文，如果出现了材料里没有的
   页码，就加一条警告提示用户核对 —— 只提示，不静默改写。
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- 配置
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OPENAI_MODEL = "deepseek-chat"
DEFAULT_OPENAI_BASE = "https://api.deepseek.com/v1"
DEFAULT_TIMEOUT = 120

# 检索不到可靠材料时用这句，不给模型发挥的余地
NO_MATERIAL_ANSWER = (
    "没有在已收录的 10 家公司年报里检索到足以回答这个问题的内容，"
    "因此不作答。\n\n"
    "可以试试：换一种问法、把指标名写全（例如把“资产规模”写成“总资产”）、"
    "或者在问题里直接点名公司（例如“中国平安”）。"
)


class LLMError(RuntimeError):
    """调用大模型失败。消息里**不含** API key。"""


@dataclass
class LLMConfig:
    """从环境变量读出来的配置。provider="none" 表示不接大模型。"""

    provider: str = "none"          # anthropic / openai / none
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    timeout: int = DEFAULT_TIMEOUT
    max_tokens: int = 1500
    temperature: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.provider != "none" and bool(self.api_key)

    @classmethod
    def from_env(cls, env: dict | None = None) -> "LLMConfig":
        """读环境变量。任何缺失都不抛错 —— 没配就退化成"不接大模型"，
        页面照样能把检索结果和引用显示出来，不至于整个用不了。
        """
        e = os.environ if env is None else env

        # 显式指定优先；否则按有没有 key 自动挑
        provider = (e.get("LLM_PROVIDER") or "").strip().lower()
        if not provider:
            if e.get("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif e.get("OPENAI_API_KEY"):
                provider = "openai"
            else:
                provider = "none"

        if provider == "anthropic":
            return cls(provider=provider,
                       model=(e.get("LLM_MODEL")
                              or e.get("ANTHROPIC_MODEL")
                              or DEFAULT_ANTHROPIC_MODEL),
                       api_key=e.get("ANTHROPIC_API_KEY", ""),
                       base_url=(e.get("ANTHROPIC_BASE_URL")
                                 or "https://api.anthropic.com").rstrip("/"),
                       timeout=int(e.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT),
                       max_tokens=int(e.get("LLM_MAX_TOKENS") or 1500))

        if provider in ("openai", "openai_compatible", "compatible"):
            return cls(provider="openai",
                       model=e.get("LLM_MODEL") or DEFAULT_OPENAI_MODEL,
                       api_key=e.get("OPENAI_API_KEY", ""),
                       base_url=(e.get("LLM_BASE_URL")
                                 or e.get("OPENAI_BASE_URL")
                                 or DEFAULT_OPENAI_BASE).rstrip("/"),
                       timeout=int(e.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT),
                       max_tokens=int(e.get("LLM_MAX_TOKENS") or 1500))

        if provider in ("none", "off", "0", "false", ""):
            return cls(provider="none")

        raise LLMError(
            f"不认识的 LLM_PROVIDER={provider!r}。可选：anthropic / openai / none")


# ---------------------------------------------------------------- 提示词
SYSTEM_PROMPT = """你是一个保险业年报问答助手。你只能依据用户提供的「材料」回答。

必须遵守：
1. **只使用材料里出现过的信息**。材料没写的，就说材料里没有，绝对不要靠记忆补全。
2. **不许编造数字、页码、科目名**。金额、比率、年份必须与材料完全一致，
   连单位（百万元 / 亿元 / 元）也要照抄，不要换算，不要四舍五入。
3. 每条事实后面标注材料编号，格式如 [1]；同时用到多条就写 [1][3]。
4. 材料之间如果数字打架（例如同一指标在不同页码口径不同），两种都列出来并说明
   分别出自哪份材料，不要替用户选一个。
5. 如果材料不足以回答问题，直接说"提供的材料不足以回答"，并说明还缺什么。
   不要勉强拼凑一个答案。

回答尽量简洁，先给结论，需要时再补一句口径或计算方式。不要输出 Markdown 表格。"""


@dataclass
class Material:
    """一条检索材料。编号从 1 开始，和提示词里的 [n] 对应。"""

    index: int
    company: str
    report_name: str
    page: int
    page_end: int | None
    section: str
    text: str

    def header(self) -> str:
        pages = (f"{self.page}页" if not self.page_end
                 or self.page_end == self.page else f"{self.page}-{self.page_end}页")
        return (f"[{self.index}] 公司：{self.company}｜报告：{self.report_name}"
                f"｜页码：{pages}｜章节：{self.section or '（未识别）'}")

    def citation(self) -> str:
        """展示用的引用块。**和 Chunk.citation() 同一格式**，直接取自元数据。"""
        pages = (f"{self.page}页" if not self.page_end
                 or self.page_end == self.page else f"{self.page}-{self.page_end}页")
        return (f"公司：{self.company}\n"
                f"报告：{self.report_name}\n"
                f"页码：{pages}\n"
                f"章节：{self.section or '（未识别）'}")


def build_materials(hits, *, max_chars: int = 1200) -> list[Material]:
    """把检索结果转成材料列表。

    Args:
        hits: RetrievedChunk 列表
        max_chars: 每条材料最多截多少字。截断是为了别把上下文撑爆 ——
            长 chunk 动辄上千字，10 条就能吃掉整个上下文预算。
            **只截尾不截头**：财报段落的关键数字常在前半部分。
    """
    out: list[Material] = []
    for i, h in enumerate(hits, start=1):
        ch = h.chunk
        text = " ".join(ch.text.split())
        if len(text) > max_chars:
            text = text[:max_chars] + " …（本段较长，已截断）"
        out.append(Material(
            index=i, company=ch.company, report_name=ch.report_name,
            page=ch.page, page_end=getattr(ch, "page_end", None),
            section=ch.section or "", text=text))
    return out


def build_user_prompt(question: str, materials: list[Material]) -> str:
    if not materials:
        return f"问题：{question}\n\n（没有任何材料）"
    blocks = "\n\n".join(f"{m.header()}\n正文：{m.text}" for m in materials)
    return (f"问题：{question}\n\n"
            f"=== 材料（共 {len(materials)} 条）===\n\n{blocks}\n\n"
            f"=== 要求 ===\n"
            f"依据上面的材料回答问题，每条事实标注材料编号如 [1]。"
            f"材料不够就说不够。")


# ---------------------------------------------------------------- HTTP
def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    """POST 一段 JSON，取回一段 JSON。抽成一个函数是为了测试时能替换掉。"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json; charset=utf-8")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:               # 4xx/5xx 才有响应体，读出来更有用
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise LLMError(f"接口返回 HTTP {e.code}。{_scrub(detail)}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"连不上模型接口（{e.reason}）。"
                       f"检查网络，或换 LLM_BASE_URL。") from e
    except TimeoutError as e:
        raise LLMError(f"调用超时（{timeout}s）。可以调大 LLM_TIMEOUT。") from e
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise LLMError(f"接口返回的不是 JSON：{_scrub(body[:300])}") from e


def _scrub(s: str) -> str:
    """把可能夹带的 key 抹掉，避免报错信息泄露密钥。"""
    return re.sub(r"(sk-|sk-ant-)[A-Za-z0-9_\-]{6,}", r"\1***", s)


# 项目根目录的 .env。**默认路径锚在项目根上，不锚在 CWD 上** ——
# 否则从别的目录调用会静默读不到（表现成"明明写了 .env 却提示未配置大模型"），
# 更糟的是可能读到别的目录里恰好的同名文件。
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def load_env_file(path=None) -> list[str]:
    """从项目根目录的 .env 读环境变量（只是图个方便，不引入 python-dotenv）。

    在 Windows 上设环境变量比较麻烦，写个 `.env` 更省事：

        ANTHROPIC_API_KEY=sk-ant-xxxx

    **只填 os.environ 里还没有的** —— 命令行上显式 `set` 的优先，不会被文件覆盖。
    返回实际设置了的变量名（**不返回值**，免得被打印进日志）。
    值不会被打印、不会被记录，也不在这个函数的返回值里。
    """
    p = Path(path) if path is not None else DEFAULT_ENV_PATH
    if not p.exists():
        return []
    set_names: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value
            set_names.append(name)
    return set_names


class LLMClient:
    """调用大模型。`transport` 可替换，测试时传假函数即可，不联网。"""

    def __init__(self, config: LLMConfig, transport=_post_json):
        self.config = config
        self._transport = transport

    def complete(self, system: str, user: str) -> str:
        """返回模型输出的纯文本。失败一律抛 LLMError。"""
        if not self.config.enabled:
            raise LLMError("没有配置大模型（缺 API key），无法生成回答。")
        if self.config.provider == "anthropic":
            return self._guarded(self._complete_anthropic, system, user)
        if self.config.provider == "openai":
            return self._guarded(self._complete_openai, system, user)
        raise LLMError(f"不支持的 provider：{self.config.provider!r}")

    @staticmethod
    def _guarded(fn, system: str, user: str) -> str:
        """统一出口：任何异常消息都先过一遍 `_scrub` 再抛。

        key 在 headers 里，正常不会出现在报错信息中；但接口回显请求体、
        或者将来换了别的 transport 时就不好说了。在**唯一出口**上兜一道，
        比指望每个调用点都记得抹要可靠 —— LLMError 的承诺就是"不含 key"。
        """
        try:
            return fn(system, user)
        except LLMError as e:
            raise LLMError(_scrub(str(e))) from e

    def _complete_anthropic(self, system: str, user: str) -> str:
        c = self.config
        payload = {
            "model": c.model,
            "max_tokens": c.max_tokens,
            "temperature": c.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        data = self._transport(f"{c.base_url}/v1/messages", payload,
                               {"x-api-key": c.api_key,
                                "anthropic-version": "2023-06-01"}, c.timeout)
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks
                       if isinstance(b, dict) and b.get("type") == "text")
        if not text:
            raise LLMError(f"接口没返回文本：{_scrub(json.dumps(data)[:300])}")
        return text.strip()

    def _complete_openai(self, system: str, user: str) -> str:
        c = self.config
        payload = {
            "model": c.model,
            "max_tokens": c.max_tokens,
            "temperature": c.temperature,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        data = self._transport(f"{c.base_url}/chat/completions", payload,
                               {"Authorization": f"Bearer {c.api_key}"},
                               c.timeout)
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"接口没返回 choices：{_scrub(json.dumps(data)[:300])}")
        text = (choices[0].get("message") or {}).get("content") or ""
        if not text:
            raise LLMError("接口返回了空的回答。")
        return text.strip()


# ------------------------------------------------------- 引用核查（防伪造）
CITED_RE = re.compile(r"\[(\d{1,2})\]")
# 只看"看起来像页码"的写法，避免把金额误判成页码
PAGE_HINT_RE = re.compile(r"(?:第\s*(\d{1,4})\s*页|(\d{1,4})\s*页)")


def extract_citations(answer: str, n_materials: int) -> list[int]:
    """取出回答里引用到的材料编号，去重保序，越界的丢掉。"""
    seen: list[int] = []
    for m in CITED_RE.finditer(answer or ""):
        i = int(m.group(1))
        if 1 <= i <= n_materials and i not in seen:
            seen.append(i)
    return seen


def find_suspicious_pages(answer: str, materials: list[Material]) -> list[int]:
    """找出正文里提到、但材料页码里没有的页码。

    这是**提示**不是拦截：模型有时会写"见第 128 页"这种话，而材料里没这页，
    那这个页码就是它自己编的。返回出来让页面标个警告，提示人工核对。
    不做静默删除 —— 删掉反而看不出模型在编。
    """
    known: set[int] = set()
    for m in materials:
        known.add(m.page)
        if m.page_end:
            known.update(range(m.page, m.page_end + 1))
    bad: list[int] = []
    for m in PAGE_HINT_RE.finditer(answer or ""):
        p = int(m.group(1) or m.group(2))
        if p not in known and p not in bad:
            bad.append(p)
    return bad


__all__ = ["LLMConfig", "LLMClient", "LLMError", "Material", "build_materials",
           "build_user_prompt", "extract_citations", "find_suspicious_pages",
           "load_env_file", "DEFAULT_ENV_PATH", "SYSTEM_PROMPT",
           "NO_MATERIAL_ANSWER"]
