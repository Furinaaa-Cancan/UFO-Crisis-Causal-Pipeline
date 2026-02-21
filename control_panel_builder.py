"""
控制面板构建器

功能：
1) 从 scraped_news.json 构建 control_topics.csv（日度话题对照）
2) 从国家RSS构建 country_controls.csv（日度国家对照）
3) 输出构建报告 control_panel_build_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparser


BASE_DIR = Path(__file__).resolve().parent
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
DEFAULT_LOOKBACK_DAYS = 120

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
            if parsed.tzinfo:
                parsed = parsed.astimezone(tz=None).replace(tzinfo=None)
            return parsed.date().isoformat()
        except Exception:
            continue

    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date().isoformat()
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
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
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
    if not COUNTRY_SOURCES_FILE.exists():
        with COUNTRY_SOURCES_FILE.open("w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
        return [s for s in default["sources"] if s.get("active", True)]

    with COUNTRY_SOURCES_FILE.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return [s for s in payload.get("sources", []) if s.get("active", True)]


def keyword_match(text: str, keywords: set[str]) -> bool:
    low = (text or "").lower()
    return any(k in low for k in keywords)


def read_existing_csv(path: Path, key_cols: Tuple[str, ...]) -> Dict[Tuple[str, ...], Dict[str, str]]:
    if not path.exists():
        return {}
    out: Dict[Tuple[str, ...], Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = tuple(row.get(k, "") for k in key_cols)
            out[key] = row
    return out


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_topic_controls(lookback_days: int) -> dict:
    if not SCRAPED_FILE.exists():
        return {"status": "blocked", "reason": "scraped_news.json 不存在", "updated": 0}

    with SCRAPED_FILE.open("r", encoding="utf-8") as f:
        scraped = json.load(f)

    rows = []
    rows.extend(scraped.get("ufo_news", []))
    rows.extend(scraped.get("crisis_news", []))
    rows.extend(scraped.get("rejected_news", []))

    today = datetime.now(timezone.utc).date()
    oldest = today - timedelta(days=lookback_days)

    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    observed_dates = set()
    considered = 0
    for r in rows:
        ds = r.get("date")
        if not ds:
            continue
        try:
            d = datetime.strptime(ds, "%Y-%m-%d").date()
        except Exception:
            continue
        if d < oldest or d > today:
            continue

        observed_dates.add(d.isoformat())
        title = normalize_text(r.get("title", ""))
        desc = normalize_text(r.get("description", ""))
        text = f"{title} {desc}".lower()
        considered += 1
        for topic, kws in TOPIC_KEYWORDS.items():
            if keyword_match(text, kws):
                counts[(d.isoformat(), topic)] += 1

    existing = read_existing_csv(TOPIC_FILE, ("date", "topic"))
    topics = sorted(TOPIC_KEYWORDS.keys())
    for date_iso in sorted(observed_dates):
        for topic in topics:
            n = counts.get((date_iso, topic), 0)
            existing[(date_iso, topic)] = {"date": date_iso, "topic": topic, "count": str(int(n))}

    out_rows = sorted(existing.values(), key=lambda x: (x["date"], x["topic"]))
    write_csv(TOPIC_FILE, ["date", "topic", "count"], out_rows)

    return {
        "status": "ok",
        "reason": "updated",
        "considered_items": considered,
        "updated": len(counts),
        "total_rows": len(out_rows),
    }


def parse_feed_items(url: str) -> Tuple[List[dict] | None, str | None]:
    resp, err = request_with_retry(url)
    if resp is None:
        return None, err

    items: List[dict] = []
    try:
        soup = BeautifulSoup(resp.content, "xml")
        entries = soup.find_all("item") or soup.find_all("entry")
        for e in entries[:MAX_ITEMS_PER_FEED]:
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
    today = datetime.now(timezone.utc).date()
    oldest = today - timedelta(days=lookback_days)

    agg: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"ufo_policy_news": 0, "crisis_index": 0})
    feed_stats = []

    for src in sources:
        country = src.get("country", "Unknown")
        name = src.get("name", "")
        url = src.get("url", "")
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
                d = datetime.strptime(it["date"], "%Y-%m-%d").date()
            except Exception:
                continue
            if d < oldest or d > today:
                continue
            text = f"{it.get('title','')} {it.get('description','')}".lower()
            key = (it["date"], country)
            _ = agg[key]  # 保留零值日期，避免只记录命中项导致面板缺列
            if keyword_match(text, UFO_KEYWORDS) and keyword_match(text, POLICY_KEYWORDS):
                agg[key]["ufo_policy_news"] += 1
            if keyword_match(text, CRISIS_KEYWORDS):
                agg[key]["crisis_index"] += 1
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

    existing = read_existing_csv(COUNTRY_FILE, ("date", "country"))
    for (date_iso, country), vals in agg.items():
        existing[(date_iso, country)] = {
            "date": date_iso,
            "country": country,
            "ufo_policy_news": str(int(vals["ufo_policy_news"])),
            "crisis_index": str(int(vals["crisis_index"])),
        }

    out_rows = sorted(existing.values(), key=lambda x: (x["date"], x["country"]))
    write_csv(COUNTRY_FILE, ["date", "country", "ufo_policy_news", "crisis_index"], out_rows)

    failures = sum(1 for s in feed_stats if s["status"] == "failed")
    if not feed_stats:
        status = "blocked"
        reason = "no_active_country_sources"
    elif failures == len(feed_stats):
        status = "blocked"
        reason = "all_country_sources_failed"
    elif failures > 0:
        status = "partial_failed"
        reason = "updated_with_partial_failures"
    else:
        status = "ok"
        reason = "updated"

    by_country = Counter(s["country"] for s in feed_stats if s["status"] == "ok")
    return {
        "status": status,
        "reason": reason,
        "sources": len(feed_stats),
        "failed_sources": failures,
        "rows_upserted": len(agg),
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": args.lookback_days,
        "topics": {"status": "skipped", "reason": "--skip-topics", "updated": 0},
        "countries": {"status": "skipped", "reason": "--skip-countries", "updated": 0},
    }

    if not args.skip_topics:
        report["topics"] = build_topic_controls(args.lookback_days)
    if not args.skip_countries:
        report["countries"] = build_country_controls(args.lookback_days)

    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== Control Panel Builder ===")
    print(f"topics: {report['topics']['status']} ({report['topics'].get('reason')})")
    print(f"countries: {report['countries']['status']} ({report['countries'].get('reason')})")
    print(f"[输出] {TOPIC_FILE}")
    print(f"[输出] {COUNTRY_FILE}")
    print(f"[输出] {REPORT_FILE}")


if __name__ == "__main__":
    main()
