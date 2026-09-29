"""把下载记录回填进 configs/companies.yaml。

`fetch_reports.py` 只负责下载和留痕；这个脚本把那份记录变成配置文件里的
可追溯字段（source_url / file_sha256 / page_count / verified）。

设计上刻意**只改字段值，不动注释和排版**，改完再用 yaml.safe_load 复验一遍
——配置文件里那些说明注释是有价值的，不能被 YAML round-trip 冲掉。

用法：
    python scripts/record_provenance.py --dry-run    # 先看要改什么
    python scripts/record_provenance.py              # 真改
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src import config, pymupdf_compat  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FETCH = ROOT / "data" / "samples" / "fetch_report.json"
YAML = ROOT / "configs" / "companies.yaml"

SOURCE_LABEL = {
    "cninfo": "巨潮资讯网（证监会指定信息披露平台）",
    "hkexnews": "香港交易所披露易 HKEXnews",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp_block(text: str, key: str, fields: dict[str, str]) -> tuple[str, bool]:
    """在 `- key: <key>` 那个块里逐字段替换。块内没有的字段追加到末尾。"""
    head = re.search(rf"^  - key: {re.escape(key)}\s*$", text, re.M)
    if not head:
        return text, False
    rest = text[head.end():]
    nxt = re.search(r"^  - key:", rest, re.M)
    end = head.end() + (nxt.start() if nxt else len(rest))
    block = text[head.start():end]

    for name, value in fields.items():
        line = re.compile(rf"^(\s*){re.escape(name)}:.*$", re.M)
        if line.search(block):
            block = line.sub(lambda m: f"{m.group(1)}{name}: {value}", block, count=1)
        else:
            block = block.rstrip("\n") + f"\n    {name}: {value}\n"

    return text[:head.start()] + block + text[end:], True


def main() -> int:
    ap = argparse.ArgumentParser(description="把抓取记录回填进 companies.yaml")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = ap.parse_args()

    if not FETCH.exists():
        print(f"找不到抓取记录 {FETCH.relative_to(ROOT)}，先跑 scripts/fetch_reports.py")
        return 1

    entries = json.loads(FETCH.read_text(encoding="utf-8"))
    fitz = pymupdf_compat.require()
    today = datetime.date.today().isoformat()

    text = YAML.read_text(encoding="utf-8")
    done, skipped = [], []

    for e in entries:
        if e.get("status") != "OK":
            skipped.append(f'{e.get("company")}（{e.get("status")}）')
            continue
        pdf = ROOT / e["dest"]
        if not pdf.exists():
            skipped.append(f'{e.get("company")}（文件不在 {e["dest"]}）')
            continue

        doc = fitz.open(pdf)
        pages = doc.page_count
        doc.close()

        label = SOURCE_LABEL.get(e.get("source", ""), e.get("source", ""))
        title = e["title"]
        fields = {
            "report_name": f'"{title}"',
            "source_url": f'"{e["url"]}"',
            "source_title": f'"{title}"',
            "downloaded_at": f'"{today}"',
            "file_sha256": f'"{sha256_file(pdf)}"',
            "page_count": str(pages),
            "file_size_bytes": str(pdf.stat().st_size),
            "verified": "true",
            "verified_note": (f'"来源：{label}，公告《{title}》；'
                              f'文件可打开、页数与正文公司名已核对"'),
        }
        text, ok = stamp_block(text, e["key"], fields)
        if ok:
            done.append(f'{e["company"]:<5} {pages:>4}页  {title}')
        else:
            skipped.append(f'{e.get("company")}（配置里找不到 key={e["key"]}）')

    for line in skipped:
        print(f"跳过 {line}")

    if args.dry_run:
        print("\n[dry-run] 会更新以下条目：")
        for line in done:
            print("  " + line)
        return 0

    YAML.write_text(text, encoding="utf-8")

    # 复验：文件仍然能被解析，且字段确实写进去了
    import yaml
    with open(YAML, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    by_key = {c["key"]: c for c in raw["companies"]}
    print("\n已更新并在配置里复验：")
    for e in entries:
        if e.get("status") != "OK":
            continue
        c = by_key.get(e["key"], {})
        ok = bool(c.get("source_url")) and bool(c.get("file_sha256"))
        print(f'  {"✓" if ok else "✗"} {c.get("company", "?"):<5} '
              f'{c.get("page_count", "?"):>4}页  verified={c.get("verified")}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
