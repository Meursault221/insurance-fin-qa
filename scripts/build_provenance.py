"""生成统一的来源清单 data/samples/final_provenance.json。

第二阶段要回答"这个数字是从哪来的"，所以 10 家公司必须用同一套来源字段。
字段分散在三个地方，本脚本把它们合成一份：

  - configs/companies.yaml        —— source_url / sha256 / 页数 / 下载时间
  - data/samples/fetch_report.json —— 抓取渠道与公告标题（只有 9 家，
    中国平安的 PDF 是更早一轮下载的，早于抓取台账存在）
  - data/raw/*.pdf                —— **实际文件**

**不是把配置值抄一遍**：sha256 / 页数 / 字节数都从本地 PDF 现算，
再和配置里的记录比对，不一致就标 FAIL 并原样列出两边的值。
抄一遍只会把配置里的错误一起复制到清单里，等于没核验。

用法：
    python scripts/build_provenance.py
    python scripts/build_provenance.py --check   # 只校验，不写文件
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src import config, pymupdf_compat  # noqa: E402

OUT = ROOT / "data" / "samples" / "final_provenance.json"
FETCH_REPORT = ROOT / "data" / "samples" / "fetch_report.json"

# 用户要求的 7 个必备字段。缺任何一个就算不合格。
REQUIRED_FIELDS = ("company", "report_year", "report_name",
                   "source_url", "file_sha256", "page_count", "local_pdf_path")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_fetch_index() -> dict[str, dict]:
    """读抓取台账，按 key 索引。文件不存在就返回空表（不报错）。"""
    if not FETCH_REPORT.exists():
        return {}
    try:
        data = json.loads(FETCH_REPORT.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    items = data if isinstance(data, list) else data.get("items", [])
    return {x["key"]: x for x in items if isinstance(x, dict) and x.get("key")}


def build_entry(prof, fitz, fetch: dict[str, dict]) -> dict:
    pdf = ROOT / "data" / "raw" / prof.expected_filename
    rec: dict = {
        "key": prof.key,
        "company": prof.company,
        "company_full": prof.company_full,
        "report_year": 2025,
        "report_name": prof.report_name,
        "source_url": prof.source_url,
        "source_channel": prof.primary_source,
        "source_title": prof.source_title,
        "downloaded_at": prof.downloaded_at,
        "file_sha256": "",
        "page_count": 0,
        "file_size_bytes": 0,
        "local_pdf_path": str(pdf.relative_to(ROOT)).replace("\\", "/"),
        "verified": prof.verified,
        "checks": {},
        "status": "OK",
    }

    # --- 文件是否存在 ---
    if not pdf.exists():
        rec["status"] = "FAIL"
        rec["checks"]["file_exists"] = False
        return rec
    rec["checks"]["file_exists"] = True

    # --- 从实际文件现算 ---
    rec["file_sha256"] = sha256_file(pdf)
    rec["file_size_bytes"] = pdf.stat().st_size
    try:
        doc = fitz.open(pdf)
        rec["page_count"] = doc.page_count
        doc.close()
    except Exception as exc:                      # noqa: BLE001
        rec["status"] = "FAIL"
        rec["checks"]["pdf_opens"] = f"{type(exc).__name__}: {exc}"
        return rec
    rec["checks"]["pdf_opens"] = True

    # --- 与配置里记的比对 ---
    sha_ok = (not prof.file_sha256) or (prof.file_sha256 == rec["file_sha256"])
    rec["checks"]["sha256_matches_config"] = sha_ok
    if prof.file_sha256 and not sha_ok:
        rec["checks"]["sha256_in_config"] = prof.file_sha256
        rec["status"] = "FAIL"

    if prof.page_count:
        pages_ok = prof.page_count == rec["page_count"]
        rec["checks"]["page_count_matches_config"] = pages_ok
        if not pages_ok:
            rec["checks"]["page_count_in_config"] = prof.page_count
            rec["status"] = "FAIL"

    if prof.file_size_bytes:
        size_ok = prof.file_size_bytes == rec["file_size_bytes"]
        rec["checks"]["size_matches_config"] = size_ok
        if not size_ok:
            rec["checks"]["size_in_config"] = prof.file_size_bytes
            rec["status"] = "FAIL"

    # --- 必备字段 ---
    missing = [f for f in REQUIRED_FIELDS
               if rec.get(f) in ("", 0, None)]
    rec["checks"]["missing_required_fields"] = missing
    if missing:
        rec["status"] = "FAIL"

    # --- 抓取台账的补充信息（平安没有，属已知情况）---
    f = fetch.get(prof.key)
    if f:
        rec["fetch_record"] = {
            "channel": f.get("source", ""),
            "announcement_title": f.get("title", ""),
            "fetch_url": f.get("url", ""),
            "detail": f.get("detail", ""),
        }
        # 台账里的 URL 应与配置一致，不一致说明两边对不上，要显式暴露
        if f.get("url") and prof.source_url and f["url"] != prof.source_url:
            rec["checks"]["fetch_url_matches_source_url"] = False
            rec["status"] = "WARN"
        else:
            rec["checks"]["fetch_url_matches_source_url"] = True
    else:
        rec["fetch_record"] = None
        rec["checks"]["in_fetch_report"] = False

    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="生成统一来源清单")
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = ap.parse_args()

    _, profiles = config.load_companies()
    fitz = pymupdf_compat.require()
    fetch = load_fetch_index()

    entries = [build_entry(p, fitz, fetch) for p in profiles]

    doc = {
        "meta": {
            "generated_by": "scripts/build_provenance.py",
            "report_year": 2025,
            "company_count": len(entries),
            "note": "sha256 / 页数 / 字节数均从 data/raw 下的实际 PDF 现算，"
                    "不是从 configs/companies.yaml 抄录；不一致会标 FAIL 并"
                    "在 checks 里给出配置中的原值。",
            "required_fields": list(REQUIRED_FIELDS),
        },
        "companies": entries,
    }

    if not args.check:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                       encoding="utf-8")

    # ---- 控制台汇总 ----
    print(f"{'公司':<6}{'年份':>5}{'页数':>6}{'本地文件':<34}{'来源':<22}状态")
    print("-" * 88)
    for e in entries:
        print(f"{e['company']:<6}{e['report_year']:>5}{e['page_count']:>6}  "
              f"{e['local_pdf_path']:<32}{e['source_channel'][:20]:<22}{e['status']}")
    print("-" * 88)

    ok = sum(1 for e in entries if e["status"] == "OK")
    warn = sum(1 for e in entries if e["status"] == "WARN")
    fail = sum(1 for e in entries if e["status"] == "FAIL")
    print(f"OK {ok} / WARN {warn} / FAIL {fail}，共 {len(entries)} 家")

    no_fetch = [e["company"] for e in entries if not e["fetch_record"]]
    if no_fetch:
        print(f"不在抓取台账中的公司（{len(no_fetch)} 家）："
              f"{'、'.join(no_fetch)} —— 其 PDF 下载早于台账建立，"
              f"来源字段取自 configs/companies.yaml，未重新下载。")

    if not args.check:
        print(f"已写入 {OUT.relative_to(ROOT)}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
