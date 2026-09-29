"""从官方披露渠道抓取年报 PDF。

只走官方接口，不碰第三方财经站：
  - A 股：巨潮资讯网 cninfo.com.cn（证监会指定信息披露平台）
          巨潮的 PDF 直链形如 http://static.cninfo.com.cn/finalpage/YYYY-MM-DD/NNNNNNNN.PDF

**这个脚本只负责下载，不负责"让它成功"。**
抓不到就把原因记下来，绝不伪造文件或来源。

用法：
    python scripts/fetch_reports.py --list                 # 看看能抓到什么，不下载
    python scripts/fetch_reports.py --keys china_life      # 抓指定公司
    python scripts/fetch_reports.py --all-a-share          # 抓全部 A 股
"""

from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " \
     "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"

CNINFO_QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_SEARCH = "http://www.cninfo.com.cn/new/information/topSearch/query"
CNINFO_STATIC = "http://static.cninfo.com.cn/"

_HEADERS = {
    "User-Agent": UA,
    "Referer": "http://www.cninfo.com.cn/",
    "X-Requested-With": "XMLHttpRequest",
}


def _post(url: str, data: dict, timeout: int = 45) -> bytes:
    body = urllib.parse.urlencode(data).encode()
    h = dict(_HEADERS, **{"Content-Type": "application/x-www-form-urlencoded"})
    req = urllib.request.Request(url, data=body, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def cninfo_org_id(stock_code: str) -> str | None:
    """按证券代码查巨潮的 orgId。查不到返回 None（不编造）。"""
    try:
        raw = _post(CNINFO_SEARCH, {"keyWord": stock_code, "maxNum": "10"})
        hits = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    for h in hits or []:
        if h.get("code") == stock_code:
            return h.get("orgId")
    return None


def _cninfo_page(stock_code: str, org: str, column: str, se: str,
                 page_num: int, category: str = "", page_size: int = 30) -> dict:
    raw = _post(CNINFO_QUERY, {
        "stock": f"{stock_code},{org}",
        "tabName": "fulltext",
        "pageSize": str(page_size),
        "pageNum": str(page_num),
        "column": column,
        "category": category,
        "searchkey": "",
        "seDate": se,
        "isHLtitle": "true",
    })
    return json.loads(raw.decode("utf-8"))


def _norm(ann: dict) -> dict:
    return {
        "title": ann.get("announcementTitle", ""),
        "url": CNINFO_STATIC + (ann.get("adjunctUrl") or "").lstrip("/"),
        "size_kb": ann.get("adjunctSize"),
        "date": ann.get("announcementTime"),
    }


def cninfo_annual_reports(stock_code: str, year: int,
                          column: str = "sse",
                          max_pages: int = 8) -> list[dict]:
    """查某公司某年度的年报类公告。

    先用 category_ndbg_szsh（年度报告）过滤。**但巨潮的类目打标并不总是可靠**
    —— 实测新华保险 2025 年年度报告没被打进该类目，过滤后一条都查不到。
    所以过滤结果里找不到完整年报时，退回到逐页扫描全部公告、按标题筛。

    返回的每项含 title / url / size_kb / date。
    """
    org = cninfo_org_id(stock_code)
    if not org:
        return []
    se = f"{year + 1}-01-01~{year + 1}-12-31"

    def _safe(page_num: int, category: str) -> list[dict]:
        try:
            payload = _cninfo_page(stock_code, org, column, se, page_num, category)
        except Exception:
            return []
        return [_norm(a) for a in (payload.get("announcements") or [])]

    tagged = _safe(1, "category_ndbg_szsh")
    if tagged and pick_full_report(tagged, year):
        return tagged

    # 兜底：翻全部公告
    seen: set[str] = set()
    out: list[dict] = []
    for pn in range(1, max_pages + 1):
        batch = _safe(pn, "")
        if not batch:
            break
        for a in batch:
            if a["url"] and a["url"] not in seen:
                seen.add(a["url"])
                out.append(a)
        if len(batch) < 30:      # 最后一页
            break
    return out


def pick_full_report(cands: list[dict], year: int) -> dict | None:
    """从候选里挑"完整年报"，排除摘要 / 英文版 / 更正公告。

    宁可返回 None 让人工判断，也不猜错一份文件。
    """
    ok = []
    for c in cands:
        t = c["title"]
        if f"{year}年年度报告" not in t:
            continue
        if any(x in t for x in ("摘要", "英文", "更正", "补充", "公告", "(H股)", "（H股）")):
            continue
        ok.append(c)
    if not ok:
        return None
    # 标题最短的那个通常是正式年报本身
    return min(ok, key=lambda c: len(c["title"]))


# --------------------------------------------------------------------------
# 港交所披露易 HKEXnews
#
# 两个接口都是 GET（POST 会返回 405）：
#   prefix.do             按代码查内部 stockId
#   titleSearchServlet.do 按 stockId + 日期 + 文档大类查公告
#
# ⚠️ market 参数必须是 "SEHK"。传 MAIN / GEM 会一条都查不到 ——
#    之前就是因此误判成"这个接口只返回 GEM 板块结果"而放弃的。
# --------------------------------------------------------------------------
HKEX_PREFIX = "https://www1.hkexnews.hk/search/prefix.do"
HKEX_SEARCH = "https://www1.hkexnews.hk/search/titleSearchServlet.do"
HKEX_ORIGIN = "https://www1.hkexnews.hk"
HKEX_REFERER = "https://www1.hkexnews.hk/search/titlesearch.xhtml"

HKEX_HEADERS = {
    "User-Agent": UA,
    "Referer": HKEX_REFERER,
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}


_hkex_opener = None
HKEX_LAST_ERROR = ""      # 最近一次失败原因，供报告记录（不静默吞掉）


def _opener():
    """带 cookie 的 opener。

    ⚠️ 必须先 GET 一次搜索页拿到 JSESSIONID / TS* cookie，
    否则 titleSearchServlet.do 会返回空的 result —— 表现为"一条都查不到"，
    很容易误判成"这家公司没披露年报"。
    """
    global _hkex_opener
    if _hkex_opener is None:
        cj = http.cookiejar.CookieJar()
        _hkex_opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(cj))
        try:
            _hkex_opener.open(urllib.request.Request(
                HKEX_REFERER + "?lang=zh",
                headers={"User-Agent": UA}), timeout=60).read()
        except Exception:
            pass
    return _hkex_opener


def _get_jsonp(url: str, timeout: int = 60) -> dict:
    """取回 callback({...}); 形式的 JSONP，剥壳后返回。"""
    global HKEX_LAST_ERROR
    try:
        req = urllib.request.Request(url, headers=HKEX_HEADERS)
        with _opener().open(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace").strip()
        if body.startswith("{"):          # titleSearchServlet 返回纯 JSON
            return json.loads(body)
        i, j = body.find("("), body.rfind(")")   # prefix.do 返回 JSONP
        if i < 0 or j < 0:
            raise ValueError(f"既不是 JSON 也不是 JSONP：{body[:120]!r}")
        return json.loads(body[i + 1:j])
    except Exception as exc:
        HKEX_LAST_ERROR = f"{type(exc).__name__}: {exc}"
        raise


def hkex_stock_id(stock_code: str) -> int | None:
    """5 位港股代码 -> 披露易内部 stockId。查不到返回 None（不猜）。"""
    code = stock_code.split(".")[0].strip().zfill(5)
    url = (f"{HKEX_PREFIX}?callback=cb&lang=EN&type=A"
           f"&name={int(code)}&market=SEHK")
    try:
        payload = _get_jsonp(url)
    except Exception:
        return None
    for s in payload.get("stockInfo") or []:
        if s.get("code") == code:
            return s.get("stockId")
    return None


def hkex_annual_reports(stock_id: int, year: int) -> list[dict]:
    """查某港股公司某年度的年报（披露易 t1code=40000 / t2code=40100）。"""
    params = {
        "sortDir": "0", "sortByOptions": "DateTime", "category": "0",
        "market": "SEHK", "stockId": str(stock_id), "documentType": "-1",
        "fromDate": f"{year + 1}0101", "toDate": f"{year + 1}1231",
        "title": "", "searchType": "1",
        "t1code": "40000",                 # 財務報表 / ESG 資料
        "t2Gcode": "-2", "t2code": "40100",  # 年報
        "rowRange": "100", "lang": "ZH",
    }
    url = HKEX_SEARCH + "?" + urllib.parse.urlencode(params)
    try:
        payload = _get_jsonp(url)
    except Exception as exc:
        print(f"    披露易查询失败：{type(exc).__name__}: {exc}")
        return []
    raw = payload.get("result")
    if isinstance(raw, str):          # result 是被二次编码的 JSON 字符串
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    out = []
    for a in raw or []:
        link = a.get("FILE_LINK") or ""
        if not link:
            continue
        out.append({
            "title": a.get("TITLE", ""),
            "url": HKEX_ORIGIN + link,
            "size": a.get("FILE_INFO", ""),
            "date": a.get("DATE_TIME", ""),
            "news_id": a.get("NEWS_ID", ""),
        })
    return out


_CN_DIGITS = "零一二三四五六七八九"
# 港股年报标题里"零"的写法五花八门：二〇二五 / 二○二五 / 二O二五 / 二零二五
_ZERO_VARIANTS = str.maketrans({"O": "零", "o": "零", "〇": "零",
                                "○": "零", "Ｏ": "零", "０": "零"})

# 不是完整年报的标题特征
_HK_EXCLUDE = ("摘要", "ESG", "環境", "社會", "管治", "可持續發展",
               "更正", "補充", "英文版", "English")


def hk_title_matches_year(title: str, year: int) -> bool:
    t = (title or "").replace(" ", "").translate(_ZERO_VARIANTS)
    cn_year = "".join(_CN_DIGITS[int(d)] for d in str(year))
    return str(year) in t or cn_year in t


def pick_hk_report(cands: list[dict], year: int) -> dict | None:
    """挑出完整年报，排除 ESG 报告、摘要、更正公告。挑不准就返回 None。"""
    ok = []
    for c in cands:
        t = (c["title"] or "").replace(" ", "")
        if not hk_title_matches_year(t, year):
            continue
        if any(x in t for x in _HK_EXCLUDE):
            continue
        ok.append(c)
    if not ok:
        return None
    return min(ok, key=lambda c: len(c["title"] or ""))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, timeout: int = 300,
             referer: str = "http://www.cninfo.com.cn/") -> tuple[bool, str]:
    """下载并做最基本的完整性检查。返回 (是否成功, 说明)。"""
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Referer": referer})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ctype = r.headers.get("Content-Type", "")
            data = r.read()
    except Exception as exc:
        return False, f"下载失败: {exc}"

    if not data:
        return False, "下载到的文件是空的"
    if not data.startswith(b"%PDF"):
        return False, f"返回的不是 PDF（Content-Type={ctype}, 前 8 字节={data[:8]!r}）"
    if len(data) < 200_000:
        return False, f"文件过小（{len(data)} 字节），不像年报"

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True, f"{len(data):,} 字节"


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="从官方渠道抓取年报 PDF")
    ap.add_argument("--keys", help="逗号分隔的配置 key")
    ap.add_argument("--all", action="store_true",
                    help="配置里所有公司（已 verified 的仍会跳过）")
    ap.add_argument("--all-a-share", action="store_true",
                    help="只抓有 A 股的公司（走巨潮）")
    ap.add_argument("--list", action="store_true", help="只列出候选，不下载")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--force", action="store_true",
                    help="连已验证的公司一起重抓（会覆盖 data/raw 里的原文件）")
    args = ap.parse_args()

    _, companies = config.load_companies()   # 返回 (meta, profiles)
    keys = [k.strip() for k in (args.keys or "").split(",") if k.strip()]
    targets, skipped = [], []
    for c in companies:
        hit = args.all or (args.all_a_share and "A" in (c.market or "")) or c.key in keys
        if not hit:
            continue
        # 已经验证通过的数据不许被覆盖 —— 除非明确 --force
        if c.verified and not args.force:
            skipped.append(c)
            continue
        targets.append(c)

    for c in skipped:
        print(f"跳过 {c.company}（已 verified，--force 可强制重抓）")

    if not targets:
        print("没有匹配的公司。用 --keys / --all / --all-a-share 指定。")
        return 1

    report = []
    for c in targets:
        # 没有 A 股的公司只能走披露易；有 A 股的走巨潮（官方且直链稳定）
        source = "hkexnews" if "A" not in (c.market or "") else "cninfo"
        code = (c.stock_codes or [""])[0].split(".")[0]
        print(f"\n=== {c.company} ({code}) [{source}] ===")

        if source == "cninfo":
            cands = cninfo_annual_reports(code, args.year)
            full = pick_full_report(cands, args.year) if cands else None
            show = lambda x: f"{x['title']}  {x['size_kb']}KB  {x['url']}"
        else:
            sid = hkex_stock_id(code)
            if sid is None:
                print("  披露易里查不到这个代码")
                report.append({"key": c.key, "company": c.company,
                               "status": "NOT_FOUND", "source": source,
                               "reason": f"披露易 prefix.do 未返回 {code} 的 stockId"})
                continue
            cands = hkex_annual_reports(sid, args.year)
            full = pick_hk_report(cands, args.year) if cands else None
            show = lambda x: f"{x['title']}  {x['size']}  {x['date']}  {x['url']}"

        if not cands:
            print("  官方接口未返回年报类公告")
            report.append({"key": c.key, "company": c.company,
                           "status": "NOT_FOUND", "source": source,
                           "reason": "官方接口未返回任何年报类公告"})
            continue

        for x in cands:
            print("   候选: " + show(x))

        if not full:
            print("  ✗ 候选里挑不出能确定的完整年报")
            report.append({"key": c.key, "company": c.company,
                           "status": "NOT_FOUND", "source": source,
                           "reason": "候选公告里挑不出完整年报（可能只有摘要/ESG/更正）"})
            continue

        print(f"  选中: {full['title']}")
        if args.list:
            report.append({"key": c.key, "company": c.company,
                           "status": "LISTED", "source": source,
                           "url": full["url"]})
            continue

        dest = RAW / c.expected_filename
        ref = HKEX_REFERER if source == "hkexnews" else "http://www.cninfo.com.cn/"
        ok, msg = download(full["url"], dest, referer=ref)
        print(f"  {'✓' if ok else '✗'} {dest.name}: {msg}")
        report.append({
            "key": c.key, "company": c.company, "source": source,
            "status": "OK" if ok else "DOWNLOAD_FAILED",
            "url": full["url"], "title": full["title"],
            "dest": str(dest.relative_to(ROOT)), "detail": msg,
            "sha256": sha256_file(dest) if ok else "",
            "size_bytes": dest.stat().st_size if ok else 0,
        })

    # 按 key 累积，不要覆盖 —— 分批抓的时候每次都重写会把前面的记录冲掉
    out = ROOT / "data" / "samples" / "fetch_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, dict] = {}
    if out.exists():
        try:
            for x in json.loads(out.read_text(encoding="utf-8")):
                merged[x["key"]] = x
        except Exception:
            pass
    for x in report:
        # 只用 --list 跑出来的条目不该盖掉之前真下载过的记录
        if x.get("status") == "LISTED" and merged.get(x["key"], {}).get("status") == "OK":
            continue
        merged[x["key"]] = x

    order = {c.key: i for i, c in enumerate(companies)}
    final = sorted(merged.values(), key=lambda x: order.get(x["key"], 999))
    out.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n抓取记录已写入 {out.relative_to(ROOT)}（累计 {len(final)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
