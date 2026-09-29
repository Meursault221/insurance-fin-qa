"""路径配置 + 公司元数据加载。

所有路径都从 PROJECT_ROOT 派生，脚本在任何工作目录下跑都不会错。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                # 原始 PDF
EXTRACTED_DIR = DATA_DIR / "extracted"    # 抽取出的结构化 JSON
CHUNKS_DIR = DATA_DIR / "chunks"          # 切分后的 chunk JSONL
SAMPLES_DIR = DATA_DIR / "samples"        # 小样本 / 演示 PDF
INDEX_DIR = PROJECT_ROOT / "index"
BM25_DIR = INDEX_DIR / "bm25"
VECTOR_DIR = INDEX_DIR / "vector"
CONFIGS_DIR = PROJECT_ROOT / "configs"
TESTS_DIR = PROJECT_ROOT / "tests"
FIXTURES_DIR = TESTS_DIR / "fixtures"
DOCS_DIR = PROJECT_ROOT / "docs"

COMPANIES_YAML = CONFIGS_DIR / "companies.yaml"


def ensure_dirs() -> None:
    """建好所有输出目录（幂等）。"""
    for d in (RAW_DIR, EXTRACTED_DIR, CHUNKS_DIR, SAMPLES_DIR,
              BM25_DIR, VECTOR_DIR, FIXTURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# 公司元数据
# --------------------------------------------------------------------------
@dataclass
class CompanyProfile:
    key: str
    company: str
    company_full: str = ""
    market: str = ""
    exchange: str = ""
    stock_codes: list[str] = None
    # 检索时用来从问题里认出公司在问谁。刻意不放歧义简称：
    # "人保"既可能指中国人保集团、也出现在"人保财险"里，只靠它无法判归属。
    aliases: list[str] = None
    report_name: str = ""
    expected_filename: str = ""
    primary_source: str = ""
    source_url: str = ""
    source_title: str = ""          # 交易所/公告平台上的原始公告标题
    downloaded_at: str = ""
    file_sha256: str = ""
    page_count: int = 0
    file_size_bytes: int = 0
    verified: bool = False
    verified_note: str = ""

    def __post_init__(self):
        if self.stock_codes is None:
            self.stock_codes = []
        if self.aliases is None:
            self.aliases = []


def load_companies(path: Optional[Path] = None) -> tuple[dict[str, Any], list[CompanyProfile]]:
    """读 configs/companies.yaml。返回 (meta, profiles)。"""
    import yaml

    path = Path(path) if path else COMPANIES_YAML
    if not path.exists():
        raise FileNotFoundError(f"找不到公司配置文件：{path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    meta = raw.get("meta", {}) or {}
    profiles = []
    for item in raw.get("companies", []) or []:
        profiles.append(
            CompanyProfile(
                key=item["key"],
                company=item["company"],
                company_full=item.get("company_full", ""),
                market=item.get("market", ""),
                exchange=item.get("exchange", ""),
                stock_codes=list(item.get("stock_codes", []) or []),
                aliases=list(item.get("aliases", []) or []),
                report_name=item.get("report_name", meta.get("report_name_default", "")),
                expected_filename=item.get("expected_filename", ""),
                primary_source=item.get("primary_source", ""),
                source_url=item.get("source_url", "") or "",
                source_title=item.get("source_title", "") or "",
                downloaded_at=str(item.get("downloaded_at", "") or ""),
                file_sha256=item.get("file_sha256", "") or "",
                page_count=int(item.get("page_count", 0) or 0),
                file_size_bytes=int(item.get("file_size_bytes", 0) or 0),
                verified=bool(item.get("verified", False)),
                verified_note=item.get("verified_note", "") or "",
            )
        )
    return meta, profiles


def get_profile(key: str, path: Optional[Path] = None) -> CompanyProfile:
    _, profiles = load_companies(path)
    for p in profiles:
        if p.key == key:
            return p
    known = ", ".join(p.key for p in profiles)
    raise KeyError(f"配置里没有 key={key!r} 的公司。可用的 key：{known}")


def guess_key_from_filename(filename: str, path: Optional[Path] = None) -> Optional[str]:
    """按文件名猜公司 key。

    匹配优先级：
      1. 完全等于 expected_filename
      2. 文件名以 "<key>_" 开头
      3. 文件名里出现 "<key>"
      4. 文件名里出现公司中文简称
    """
    stem = Path(filename).stem.lower()
    _, profiles = load_companies(path)

    for p in profiles:
        if p.expected_filename and Path(p.expected_filename).stem.lower() == stem:
            return p.key
    for p in profiles:
        if stem.startswith(p.key.lower() + "_") or stem == p.key.lower():
            return p.key
    for p in profiles:
        if p.key.lower() in stem:
            return p.key
    for p in profiles:
        if p.company and p.company in filename:
            return p.key
    return None


def guess_year_from_filename(filename: str, default: int = 2025) -> int:
    """从文件名里抓 4 位年份。抓不到就用 default。"""
    m = re.search(r"(20\d{2})", filename)
    return int(m.group(1)) if m else default


def guess_report_type_from_filename(filename: str) -> str:
    n = filename.lower()
    if any(k in n for k in ("semi", "half", "mid", "半年", "中期", "中报")):
        return "semi_annual"
    return "annual"


def guess_report_name(report_year: int, report_type: str, is_hk: bool = False) -> str:
    if report_type == "semi_annual":
        return f"{report_year}年半年度报告"
    return f"{report_year}年報" if is_hk else f"{report_year}年年度报告"
