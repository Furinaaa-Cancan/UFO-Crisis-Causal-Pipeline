"""
控制面板构建器

功能：
1) 从 scraped_news.json 构建 control_topics.csv（日度话题对照）
2) 从国家RSS构建 country_controls.csv（日度国家对照）
3) 输出构建报告 control_panel_build_report.json
"""
# pyre-ignore-all-errors

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparser


BASE_DIR = Path(__file__).resolve().parent  # type: ignore
DATA_DIR = BASE_DIR / "data"
CONTROL_DIR = DATA_DIR / "control_panels"
SCRAPED_FILE = DATA_DIR / "scraped_news.json"
TOPIC_FILE = CONTROL_DIR / "control_topics.csv"
COUNTRY_FILE = CONTROL_DIR / "country_controls.csv"
COUNTRY_SOURCES_FILE = CONTROL_DIR / "country_sources.json"
REPORT_FILE = CONTROL_DIR / "control_panel_build_report.json"

REQUEST_TIMEOUT = 20
REQUEST_RETRIES = 3
MAX_ITEMS_PER_FEED = 120
# Use a long-horizon default so DID/Synth can estimate on historical windows.
DEFAULT_LOOKBACK_DAYS = 3650

TOPIC_KEYWORDS = {
    "sports": {
        "nfl", "nba", "mlb", "nhl", "soccer", "olympics", "championship", "super bowl", "fifa",
    },
    "entertainment": {
        "movie", "film", "tv", "television", "music", "celebrity", "hollywood", "award", "netflix",
    },
    "weather": {
        "weather", "storm", "hurricane", "tornado", "flood", "heatwave", "blizzard", "rainfall", "wildfire",
    },
    "technology": {
        "ai", "artificial intelligence", "chip", "software", "technology", "apple", "google", "microsoft", "openai",
    },
}

UFO_KEYWORDS = {
    "ufo", "uap", "alien", "extraterrestrial", "non-human", "nonhuman", "unidentified aerial",
}

POLICY_KEYWORDS = {
    "government", "congress", "senate", "house", "pentagon", "defense", "official", "ministry", "parliament",
    "doj", "fbi", "hearing", "committee", "white house",
}

CRISIS_KEYWORDS = {
    "indictment", "scandal", "investigation", "arrest", "impeach", "subpoena", "corruption", "charges",
    "special counsel", "grand jury", "obstruction", "classified",
}


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def parse_feed_date(date_str: str) -> str | None:
    if not date_str:
        return None
    for parser_fn in (
        lambda s: parsedate_to_datetime(s),
        lambda s: dtparser.parse(s, fuzzy=True),
    ):
        try:
            parsed = parser_fn(date_str)
            if parsed.tzinfo:  # type: ignore
                parsed = parsed.astimezone(tz=None).replace(tzinfo=None)  # type: ignore
            return parsed.date().isoformat()  # type: ignore
        except Exception:
            continue

    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date().isoformat()  # type: ignore
    except Exception:
        return None


def request_with_retry(url: str) -> Tuple[requests.Response | None, str | None]:
    last_error = None
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)  # type: ignore
            if resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                continue
            resp.raise_for_status()
            return resp, None
        except Exception as e:
            last_error = str(e)
            if attempt < REQUEST_RETRIES:
                continue
    return None, last_error or "unknown"


def load_country_sources() -> List[dict]:
    default = {
        "sources": [
            {
                "country": "US",
                "name": "NYT Politics",
                "url": "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
                "active": True,
            },
            {
                "country": "UK",
                "name": "BBC Politics",
                "url": "https://feeds.bbci.co.uk/news/politics/rss.xml",
                "active": True,
            },
            {
                "country": "Canada",
                "name": "CBC Politics",
                "url": "https://www.cbc.ca/webfeed/rss/rss-politics",
                "active": True,
            },
            {
                "country": "Australia",
                "name": "ABC Australia Politics",
                "url": "https://www.abc.net.au/news/feed/51120/rss.xml",
                "active": True,
            },
        ]
    }
    if not COUNTRY_SOURCES_FILE.exists():  # type: ignore
        with COUNTRY_SOURCES_FILE.open("w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)  # type: ignore
        return [s for s in default["sources"] if s.get("active", True)]  # type: ignore

    with COUNTRY_SOURCES_FILE.open("r", encoding="utf-8") as f:
        payload = json.load(f)  # type: ignore
    return [s for s in payload.get("sources", []) if s.get("active", True)]  # type: ignore


def keyword_match(text: str, keywords: set[str]) -> bool:
    low = (text or "").lower()

    def contains_kw(kw: str) -> bool:
        k = (kw or "").lower().strip()
        if not k:
            return False
        # Avoid substring false positives (e.g. "ai" matching "said").
        if re.fullmatch(r"[a-z0-9]+", k):
            return re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", low) is not None
        return k in low

    return any(contains_kw(k) for k in keywords)


def read_existing_csv(path: Path, key_cols: Tuple[str, ...]) -> Dict[Tuple[str, ...], Dict[str, str]]:
    if not path.exists():  # type: ignore
        return {}
    out: Dict[Tuple[str, ...], Dict[str, str]] = {}  # type: ignore
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)  # type: ignore
        for row in reader:
            key = tuple(row.get(k, "") for k in key_cols)  # type: ignore
            out[key] = row  # type: ignore
    return out


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)  # type: ignore
        writer.writeheader()
        writer.writerows(rows)  # type: ignore


def build_date_grid(oldest, today) -> List[str]:
    if oldest > today:
        return []
    n_days = (today - oldest).days + 1  # type: ignore
    return [(oldest + timedelta(days=i)).isoformat() for i in range(n_days)]  # type: ignore


def build_topic_controls(lookback_days: int) -> dict:
    if not SCRAPED_FILE.exists():  # type: ignore
        return {"status": "blocked", "reason": "scraped_news.json 不存在", "updated": 0}

    with SCRAPED_FILE.open("r", encoding="utf-8") as f:
        scraped = json.load(f)  # type: ignore

    rows = []
    rows.extend(scraped.get("ufo_news", []))  # type: ignore
    rows.extend(scraped.get("crisis_news", []))  # type: ignore
    rows.extend(scraped.get("rejected_news", []))  # type: ignore

    today = datetime.now(timezone.utc).date()  # type: ignore
    oldest = today - timedelta(days=lookback_days)  # type: ignore
    grid_dates = build_date_grid(oldest, today)

    counts: Dict[Tuple[str, str], int] = defaultdict(int)  # type: ignore
    considered = 0
    for r in rows:
        ds = r.get("date")  # type: ignore
        if not ds:
            continue
        try:
            d = datetime.strptime(ds, "%Y-%m-%d").date()  # type: ignore
        except Exception:
            continue
        if d < oldest or d > today:
            continue

        title = normalize_text(r.get("title", ""))  # type: ignore
        desc = normalize_text(r.get("description", ""))  # type: ignore
        text = f"{title} {desc}".lower()
        considered += 1
        for topic, kws in TOPIC_KEYWORDS.items():  # type: ignore
            if keyword_match(text, kws):
                counts[(d.isoformat(), topic)] += 1  # type: ignore

    topics = sorted(TOPIC_KEYWORDS.keys())  # type: ignore
    out_rows = []
    for date_iso in grid_dates:
        for topic in topics:
            n = counts.get((date_iso, topic), 0)  # type: ignore
            out_rows.append({"date": date_iso, "topic": topic, "count": str(int(n))})  # type: ignore
    write_csv(TOPIC_FILE, ["date", "topic", "count"], out_rows)

    return {
        "status": "ok",
        "reason": "zero_filled_full_date_grid",
        "considered_items": considered,
        "updated_nonzero_cells": len(counts),
        "grid_days": len(grid_dates),
        "topics": len(topics),
        "total_rows": len(out_rows),
    }


def parse_feed_items(url: str) -> Tuple[List[dict] | None, str | None]:
    resp, err = request_with_retry(url)
    if resp is None:
        return None, err

    items: List[dict] = []  # type: ignore
    try:
        soup = BeautifulSoup(resp.content, "xml")
        entries = soup.find_all("item") or soup.find_all("entry")
        for e in entries[:MAX_ITEMS_PER_FEED]:  # type: ignore
            title_tag = e.find("title")
            date_tag = e.find("pubDate") or e.find("published") or e.find("updated") or e.find("dc:date")
            desc_tag = e.find("description") or e.find("summary") or e.find("content")
            title = normalize_text(title_tag.get_text(" ", strip=True) if title_tag else "")
            raw_date = normalize_text(date_tag.get_text(" ", strip=True) if date_tag else "")
            date_iso = parse_feed_date(raw_date)
            desc = normalize_text(desc_tag.get_text(" ", strip=True) if desc_tag else "")
            if not title or not date_iso:
                continue
            items.append({"title": title, "description": desc, "date": date_iso})
    except Exception as e:
        return None, f"parse error: {e}"

    return items, None


def build_country_controls(lookback_days: int) -> dict:
    sources = load_country_sources()
    today = datetime.now(timezone.utc).date()  # type: ignore
    oldest = today - timedelta(days=lookback_days)  # type: ignore
    grid_dates = build_date_grid(oldest, today)
    countries = sorted({str(src.get("country", "Unknown")) for src in sources})  # type: ignore

    agg: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"ufo_policy_news": 0, "crisis_index": 0})  # type: ignore
    feed_stats = []

    for src in sources:
        country = src.get("country", "Unknown")  # type: ignore
        name = src.get("name", "")  # type: ignore
        url = src.get("url", "")  # type: ignore
        items, err = parse_feed_items(url)
        if items is None:
            feed_stats.append({
                "country": country,
                "source": name,
                "url": url,
                "status": "failed",
                "error": err,
                "items": 0,
            })
            continue

        kept = 0
        for it in items:
            try:
                d = datetime.strptime(it["date"], "%Y-%m-%d").date()  # type: ignore
            except Exception:
                continue
            if d < oldest or d > today:
                continue
            text = f"{it.get('title','')} {it.get('description','')}".lower()  # type: ignore
            key = (it["date"], country)  # type: ignore
            _ = agg[key]  # 保留零值日期，避免只记录命中项导致面板缺列  # type: ignore
            if keyword_match(text, UFO_KEYWORDS) and keyword_match(text, POLICY_KEYWORDS):
                agg[key]["ufo_policy_news"] += 1  # type: ignore
            if keyword_match(text, CRISIS_KEYWORDS):
                agg[key]["crisis_index"] += 1  # type: ignore
            kept += 1

        feed_stats.append({
            "country": country,
            "source": name,
            "url": url,
            "domain": urlsplit(url).netloc.lower(),
            "status": "ok",
            "error": None,
            "items": kept,
        })

    out_rows = []
    for date_iso in grid_dates:
        for country in countries:
            vals = agg.get((date_iso, country), {"ufo_policy_news": 0, "crisis_index": 0})  # type: ignore
            out_rows.append({
                "date": date_iso,
                "country": country,
                "ufo_policy_news": str(int(vals["ufo_policy_news"])),  # type: ignore
                "crisis_index": str(int(vals["crisis_index"])),  # type: ignore
            })
    write_csv(COUNTRY_FILE, ["date", "country", "ufo_policy_news", "crisis_index"], out_rows)

    failures = sum(1 for s in feed_stats if s["status"] == "failed")  # type: ignore
    if not feed_stats:
        status = "blocked"
        reason = "no_active_country_sources"
    elif failures == len(feed_stats):
        status = "ok"
        reason = "all_country_sources_failed_zero_filled_grid"
    elif failures > 0:
        status = "partial_failed"
        reason = "updated_with_partial_failures_zero_filled_grid"
    else:
        status = "ok"
        reason = "updated_zero_filled_full_date_grid"

    by_country = Counter(s["country"] for s in feed_stats if s["status"] == "ok")  # type: ignore
    return {
        "status": status,
        "reason": reason,
        "sources": len(feed_stats),
        "failed_sources": failures,
        "rows_with_nonzero_updates": len(agg),
        "grid_days": len(grid_dates),
        "countries": len(countries),
        "total_rows": len(out_rows),
        "country_source_success": dict(by_country),
        "feed_stats": feed_stats,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建 control topics + country controls")
    p.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    p.add_argument("--skip-topics", action="store_true")
    p.add_argument("--skip-countries", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # type: ignore
        "lookback_days": args.lookback_days,
        "topics": {"status": "skipped", "reason": "--skip-topics", "updated": 0},
        "countries": {"status": "skipped", "reason": "--skip-countries", "updated": 0},
    }

    if not args.skip_topics:
        report["topics"] = build_topic_controls(args.lookback_days)  # type: ignore
    if not args.skip_countries:
        report["countries"] = build_country_controls(args.lookback_days)  # type: ignore

    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)  # type: ignore

    print("=== Control Panel Builder ===")
    print(f"topics: {report['topics']['status']} ({report['topics'].get('reason')})")  # type: ignore
    print(f"countries: {report['countries']['status']} ({report['countries'].get('reason')})")  # type: ignore
    print(f"[输出] {TOPIC_FILE}")
    print(f"[输出] {COUNTRY_FILE}")
    print(f"[输出] {REPORT_FILE}")


if __name__ == "__main__":
    main()
