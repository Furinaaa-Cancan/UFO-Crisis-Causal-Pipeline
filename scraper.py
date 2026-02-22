"""
新闻抓取与关联分析器（严格真实性模式）
流程：抓取 -> 规范化 -> 真实性评分 -> 跨来源佐证 -> 严格过滤 -> 关联分析
"""
# pyre-ignore-all-errors
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed  # type: ignore
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit  # type: ignore

import requests  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
from dateutil import parser as dtparser  # type: ignore
from rich import box  # type: ignore
from rich.console import Console  # type: ignore
from rich.progress import Progress, SpinnerColumn, TextColumn  # type: ignore
from rich.table import Table  # type: ignore

console = Console()

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
SCRAPED_FILE = os.path.join(OUTPUT_DIR, "scraped_news.json")
SOURCES_FILE = os.path.join(OUTPUT_DIR, "sources.json")
EVENTS_V2_FILE = os.path.join(OUTPUT_DIR, "events_v2.json")
OFFICIAL_LEAD_DIAG_FILE = os.path.join(OUTPUT_DIR, "official_lead_event_candidates.json")

REQUEST_TIMEOUT = 20
REQUEST_RETRIES = 3
LOOKBACK_DAYS_DEFAULT = 120
FULL_HISTORY_LOOKBACK_DAYS = 36500
WINDOW_DAYS_DEFAULT = 60
STRICT_MIN_SCORE = 65
LENIENT_MIN_SCORE = 45
BALANCED_MIN_SCORE = 58
MAX_WORKERS_DEFAULT = 6
MAX_ITEMS_PER_SOURCE = 30

POLICY_STRICT = "strict"
POLICY_STRICT_BALANCED = "strict-balanced"
POLICY_LENIENT = "lenient"
POLICY_CHOICES = [POLICY_STRICT, POLICY_STRICT_BALANCED, POLICY_LENIENT]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}
REDDIT_HEADERS = {
    "User-Agent": "UFO-Crisis-Analyzer/2.0 (research tool)"
}

AGGREGATOR_SOURCE_HINTS = ("google news",)
OFFICIAL_SOURCE_HINTS = (
    "white house",
    "pentagon",
    "department of defense",
    "dod",
    "state department",
    "congress",
    "senate",
    "house",
    "doj",
    "fbi",
    "cia",
    "nasa",
    "aaro",
)
TRACKING_QUERY_KEYS = {
    "fbclid", "gclid", "oc", "ref", "source", "utm_source",
    "utm_medium", "utm_campaign", "utm_term", "utm_content",
}

TITLE_SPAM_PATTERNS = [
    re.compile(r"!{3,}"),
    re.compile(r"\?{3,}"),
    re.compile(r"^[A-Z0-9\W]{16,}$"),
]

CLAIM_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "about",
    "after", "over", "under", "amid", "says", "said", "will", "would",
    "could", "should", "have", "has", "had", "his", "her", "their",
    "its", "who", "what", "when", "where", "why", "how", "than", "then",
    "news", "report", "reports", "official", "officials", "government",
    "president", "white", "house", "pentagon", "new", "latest", "update",
}

AUTH_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

UFO_KEYWORDS = [
    "ufo", "uap", "alien", "extraterrestrial", "flying saucer",
    "unidentified aerial", "unidentified flying", "disclosure",
    "non-human", "nonhuman", "roswell", "area 51", "crash retrieval",
    "bob lazar", "grusch", "elizondo", "aaro", "aatip",
    "non-human intelligence", "ufo files", "uap task force",
    "ufo hearing", "uap hearing", "declassified ufo",
    "pentagon ufo", "navy ufo", "senate ufo", "congress ufo",
    "whistleblower ufo", "tic tac ufo", "nimitz incident",
    "gimbal ufo", "go fast ufo", "skinwalker",
    "unidentified domain awareness", "transmedium",
    "unidentified anomalous", "novel aerial", "aerial phenomena",
    "unexplained aerial", "unidentified objects", "unknown aerial",
    "recovered materials", "non-human craft", "reverse engineering",
]

CRISIS_KEYWORDS = [
    "indictment", "impeach", "scandal", "arrest", "investigation",
    "mueller", "epstein", "classified documents", "whistleblower",
    "coverup", "corruption", "resign", "crisis", "special counsel",
    "grand jury", "subpoena", "contempt of congress", "obstruction",
    "hush money", "felony", "federal charges", "pardon", "abuse of power",
    "emoluments", "insurrection", "sedition", "bribery",
    "perjury", "money laundering", "obstruction of justice",
]

CRISIS_HARD_SIGNAL_KEYWORDS = [
    "indictment", "impeach", "scandal", "special counsel", "grand jury",
    "subpoena", "federal charges", "classified documents", "whistleblower",
    "obstruction", "obstruction of justice", "insurrection", "sedition",
    "bribery", "perjury", "money laundering", "abuse of power",
    "emoluments", "coverup", "corruption", "epstein",
]

OFFICIAL_ACTION_KEYWORDS = [
    "supreme court", "appeals court", "court order", "court ruling", "ruling",
    "ruled", "rules", "strikes down", "struck down", "overturn", "overturned",
    "injunction", "injunctive", "enjoined", "blocked", "veto", "signed into law",
    "executive order", "hearing", "committee vote", "vote to", "passed bill",
    "filed lawsuit", "lawsuit", "charged", "convicted", "sentenced", "plea deal",
]

CRISIS_TOPIC_KEYWORDS = [
    "tariff", "tariffs", "immigration", "border", "epstein", "iran", "russia",
    "ukraine", "china", "israel", "gaza", "tax", "budget", "deficit",
    "inflation", "election", "campaign", "supreme court", "doj", "fbi",
]

POLITICAL_CONTEXT_KEYWORDS = [
    "president", "white house", "congress", "senate", "house",
    "doj", "department of justice", "fbi", "federal", "supreme court",
    "attorney general", "administration", "campaign", "election",
    "impeach", "indictment", "subpoena", "grand jury",
    "democrat", "democrats", "republican", "republicans",
    "trump", "biden", "obama", "governor", "mayor", "parliament",
    "minister", "cabinet", "state department", "pentagon",
]

NATIONAL_POLITICAL_CONTEXT_KEYWORDS = [
    "president", "white house", "congress", "senate", "house",
    "doj", "department of justice", "fbi", "federal", "supreme court",
    "campaign", "election", "state department",
    "pentagon", "trump", "biden", "obama",
]

OPINION_TITLE_HINTS = [
    "opinion:", "analysis:", "editorial:", "commentary:", "guest essay:",
]

TITLE_NATIONAL_POLITICAL_KEYWORDS = [
    "president", "white house", "congress", "senate", "house", "supreme court",
    "doj", "department of justice", "fbi", "federal", "state department",
    "pentagon", "campaign", "election",
    "trump", "biden", "obama", "democrat", "republican",
]

POLICY_CONFIGS = {
    POLICY_STRICT: {
        "min_score": STRICT_MIN_SCORE,
        "enforce_crisis_title_actor": True,
        "enforce_crisis_title_hard_signal": True,
        "enforce_ufo_title_signal": True,
        "enforce_opinion_filter": True,
        "enforce_roundup_filter": True,
        "strict_source_rules": True,
        "require_crisis_national_context": True,
        "require_crisis_hard_signal": True,
    },
    POLICY_STRICT_BALANCED: {
        "min_score": 55,
        "enforce_crisis_title_actor": False,
        "enforce_crisis_title_hard_signal": False,
        "enforce_ufo_title_signal": False,
        "enforce_opinion_filter": True,
        "enforce_roundup_filter": True,
        "strict_source_rules": True,
        "require_crisis_national_context": True,
        "require_crisis_hard_signal": False,
    },
    POLICY_LENIENT: {
        "min_score": LENIENT_MIN_SCORE,
        "enforce_crisis_title_actor": False,
        "enforce_crisis_title_hard_signal": False,
        "enforce_ufo_title_signal": False,
        "enforce_opinion_filter": False,
        "enforce_roundup_filter": False,
        "strict_source_rules": False,
        "require_crisis_national_context": False,
        "require_crisis_hard_signal": False,
    },
}


def _parse_item_date(value):
    if not value:
        return None
    return parse_iso_date(str(value))


def summarize_date_span(items):
    dates = []
    for item in items:
        d = _parse_item_date(item.get("date"))  # type: ignore
        if d is not None:
            dates.append(d)
    if not dates:
        return {
            "min_date": None,
            "max_date": None,
            "span_days": 0,
            "dated_items": 0,
        }
    lo = min(dates)
    hi = max(dates)
    return {
        "min_date": lo.isoformat(),  # type: ignore
        "max_date": hi.isoformat(),  # type: ignore
        "span_days": (hi - lo).days + 1,
        "dated_items": len(dates),
    }


def build_coverage_audit(
    raw_items,
    unique_items,
    candidates,
    accepted_events,
    rejected_items,
    source_stats,
    lookback_days,
):
    today = datetime.now(timezone.utc).date()  # type: ignore
    expected_start = (today - timedelta(days=lookback_days)).isoformat()  # type: ignore
    expected_end = today.isoformat()  # type: ignore

    reason_counter = Counter()
    stale_count = 0
    for row in rejected_items:
        reason = str(row.get("reason", ""))  # type: ignore
        if not reason:
            continue
        for token in reason.split(","):
            token = token.strip()
            if not token:
                continue
            reason_counter[token] += 1  # type: ignore
            if token == "stale_item_outside_lookback":
                stale_count += 1

    by_source = {}
    for item in raw_items:
        src = item.get("source", "unknown")  # type: ignore
        rec = by_source.setdefault(src, {"count": 0, "dates": []})  # type: ignore
        rec["count"] += 1  # type: ignore
        d = _parse_item_date(item.get("date"))  # type: ignore
        if d is not None:
            rec["dates"].append(d)  # type: ignore

    freshness_cutoff = today - timedelta(days=3)  # type: ignore
    source_windows = []
    fresh_sources = 0
    for row in source_stats:
        source_name = row.get("source", "unknown")  # type: ignore
        dates = sorted(by_source.get(source_name, {}).get("dates", []))
        if dates:
            min_date = dates[0]
            max_date = dates[-1]
            if max_date >= freshness_cutoff:
                fresh_sources += 1
            source_windows.append({
                "source": source_name,
                "status": row.get("status"),  # type: ignore
                "item_count": int(row.get("item_count", 0) or 0),  # type: ignore
                "min_date": min_date.isoformat(),  # type: ignore
                "max_date": max_date.isoformat(),  # type: ignore
                "span_days": (max_date - min_date).days + 1,
                "has_recent_items": bool(max_date >= freshness_cutoff),
                "used_fallback": bool(row.get("used_fallback", False)),  # type: ignore
            })
        else:
            source_windows.append({
                "source": source_name,
                "status": row.get("status"),  # type: ignore
                "item_count": int(row.get("item_count", 0) or 0),  # type: ignore
                "min_date": None,
                "max_date": None,
                "span_days": 0,
                "has_recent_items": False,
                "used_fallback": bool(row.get("used_fallback", False)),  # type: ignore
            })

    source_windows.sort(key=lambda x: ((x["status"] != "ok"), x["source"]))  # type: ignore
    available_sources = sum(1 for r in source_stats if r.get("status") != "failed")  # type: ignore
    coverage_confidence = (
        "high"
        if (available_sources >= max(1, int(len(source_stats) * 0.9)) and fresh_sources >= max(1, int(len(source_stats) * 0.7)))
        else "medium"
        if (available_sources >= max(1, int(len(source_stats) * 0.75)) and fresh_sources >= max(1, int(len(source_stats) * 0.5)))
        else "low"
    )

    return {
        "window": {
            "expected_start": expected_start,
            "expected_end": expected_end,
            "lookback_days": lookback_days,
        },
        "date_spans": {
            "raw_items": summarize_date_span(raw_items),
            "unique_items": summarize_date_span(unique_items),
            "candidates": summarize_date_span(candidates),
            "accepted_events": summarize_date_span(accepted_events),
        },
        "rejections": {
            "stale_item_outside_lookback": stale_count,
            "top_reasons": [
                {"reason": reason, "count": count}
                for reason, count in reason_counter.most_common(10)
            ],
        },
        "source_window_summary": {
            "sources_total": len(source_stats),
            "sources_available": available_sources,
            "sources_with_recent_items": fresh_sources,
            "coverage_confidence": coverage_confidence,
        },
        "source_windows": source_windows,
    }


def print_coverage_summary(coverage):
    summary = coverage.get("source_window_summary", {})  # type: ignore
    spans = coverage.get("date_spans", {})  # type: ignore
    accepted_span = spans.get("accepted_events", {})  # type: ignore
    raw_span = spans.get("raw_items", {})  # type: ignore

    t = Table(title="时间覆盖审计", box=box.SIMPLE)  # type: ignore
    t.add_column("指标", width=34)
    t.add_column("值", width=50)
    t.add_row(
        "目标窗口",
        f"{coverage.get('window', {}).get('expected_start')} ~ {coverage.get('window', {}).get('expected_end')}",  # type: ignore
    )
    t.add_row(
        "原始条目时间范围",
        f"{raw_span.get('min_date')} ~ {raw_span.get('max_date')} ({raw_span.get('span_days', 0)}天)",  # type: ignore
    )
    t.add_row(
        "通过事件时间范围",
        f"{accepted_span.get('min_date')} ~ {accepted_span.get('max_date')} ({accepted_span.get('span_days', 0)}天)",  # type: ignore
    )
    t.add_row(
        "来源近期覆盖",
        (
            f"{summary.get('sources_with_recent_items', 0)}/{summary.get('sources_total', 0)} "
            f"(confidence={summary.get('coverage_confidence', 'low')})"
        ),
    )
    t.add_row(
        "超窗剔除数",
        str(coverage.get("rejections", {}).get("stale_item_outside_lookback", 0)),  # type: ignore
    )
    console.print(t)  # type: ignore


def normalize_text(text):
    return " ".join((text or "").split())


def strip_html(text):
    if not text:
        return ""
    # Avoid feeding plain URLs/paths into BeautifulSoup to reduce parser warnings.
    if "<" not in text and ">" not in text:
        return normalize_text(text)
    return normalize_text(BeautifulSoup(text, "html.parser").get_text(" ", strip=True))  # type: ignore


def parse_iso_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()  # type: ignore
    except ValueError:
        return None


def parse_iso_datetime(datetime_str):
    if not datetime_str:
        return None
    raw = str(datetime_str).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def parse_feed_datetime(date_str):
    """解析 RSS/Atom 日期；返回 (YYYY-MM-DD, published_at_iso_utc, parsed_ok)"""
    if not date_str:
        return None, None, False

    for parser_fn in (
        lambda s: parsedate_to_datetime(s),
        lambda s: dtparser.parse(s, fuzzy=True),
    ):
        try:  # type: ignore
            parsed = parser_fn(date_str)  # type: ignore
            if parsed.tzinfo is None:  # type: ignore
                parsed_utc = parsed.replace(tzinfo=timezone.utc)  # type: ignore
            else:
                parsed_utc = parsed.astimezone(timezone.utc)  # type: ignore
            return parsed_utc.date().isoformat(), parsed_utc.isoformat(), True  # type: ignore
        except Exception:
            continue

    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()  # type: ignore
        return d.isoformat(), None, True  # type: ignore
    except Exception:
        return None, None, False


def parse_feed_date(date_str):
    """兼容旧逻辑：仅返回日期与解析状态"""
    date_iso, _, parsed_ok = parse_feed_datetime(date_str)
    return date_iso, parsed_ok


def canonicalize_url(url):
    if not url:
        return ""

    url = url.strip()
    try:
        p = urlsplit(url)
    except Exception:
        return url

    if not p.scheme or not p.netloc:
        return url

    host = p.netloc.lower()  # type: ignore
    if host.startswith("www."):
        host = host[4:]  # type: ignore

    clean_query = []
    for k, v in parse_qsl(p.query, keep_blank_values=True):
        key = k.lower()
        if key.startswith("utm_") or key in TRACKING_QUERY_KEYS:
            continue
        clean_query.append((k, v))

    query = urlencode(clean_query, doseq=True)
    path = p.path.rstrip("/") or "/"
    return urlunsplit((p.scheme.lower(), host, path, query, ""))


def keyword_hits(text, keywords):
    text = (text or "").lower()

    def contains_kw(kw):
        k = (kw or "").lower().strip()
        if not k:
            return False
        # Avoid substring false positives (e.g. "war" in "forward", "ai" in "said").
        if re.fullmatch(r"[a-z0-9]+", k):
            return re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", text) is not None
        return k in text

    return [kw for kw in keywords if contains_kw(kw)]


def source_is_aggregator(source_name):
    lowered = (source_name or "").lower()
    return any(h in lowered for h in AGGREGATOR_SOURCE_HINTS)


def source_is_official(source_name):
    lowered = (source_name or "").lower()
    return any(h in lowered for h in OFFICIAL_SOURCE_HINTS)


def source_is_trusted(item):
    auth = item.get("authenticity", {})  # type: ignore
    if auth.get("is_aggregator"):  # type: ignore
        return False
    return item.get("source_type") == "rss" and item.get("weight", 1) >= 2  # type: ignore


def looks_spammy_title(title):
    title = title or ""
    return any(p.search(title) for p in TITLE_SPAM_PATTERNS)


def looks_opinion_or_commentary(title):
    t = (title or "").lower()
    if any(h in t for h in OPINION_TITLE_HINTS):
        return True
    # Guardian等媒体常见专栏形式：标题 + " | 作者名"
    return " | " in title


def looks_roundup_or_mixed_headline(title):
    t = (title or "").lower()
    if ". and," in t or "; and " in t:
        return True
    if t.count(" and, ") >= 1:
        return True
    return False


def extract_claim_tokens(title):
    words = re.findall(r"[a-z0-9]+", (title or "").lower())
    filtered = [w for w in words if len(w) > 2 and w not in CLAIM_STOPWORDS]  # type: ignore
    if not filtered:
        filtered = words[:6]  # type: ignore
    deduped = []
    seen = set()
    for w in filtered:
        if w in seen:
            continue
        seen.add(w)  # type: ignore
        deduped.append(w)
    return deduped[:8]  # type: ignore


def claim_fingerprint(item):
    tokens = extract_claim_tokens(item.get("title", ""))  # type: ignore
    basis = f"{item.get('category','')}-{' '.join(tokens)}"  # type: ignore
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]  # type: ignore
    return digest, tokens


def confidence_label(score):
    if score >= 80:
        return "HIGH"
    if score >= 60:
        return "MEDIUM"
    return "LOW"


def resolve_policy(policy_name):
    policy_name = (policy_name or POLICY_STRICT).lower()
    if policy_name not in POLICY_CONFIGS:
        policy_name = POLICY_STRICT
    return policy_name, POLICY_CONFIGS[policy_name]  # type: ignore


def load_source_config():
    if not os.path.exists(SOURCES_FILE):  # type: ignore
        raise FileNotFoundError(SOURCES_FILE)

    with open(SOURCES_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)  # type: ignore

    all_sources = config.get("sources", [])  # type: ignore
    sources = [s for s in all_sources if s.get("active", True)]  # type: ignore
    console.print(f"[dim]已加载 {len(sources)} 个活跃来源[/dim]")  # type: ignore
    return config.get("_meta", {}), all_sources, sources  # type: ignore


def load_sources():
    _, _, sources = load_source_config()
    return sources


def classify_source_error(error_text):
    low = (error_text or "").lower()
    if not low:
        return "none"
    if "ssl" in low or "tls" in low or "certificate" in low:
        return "ssl_tls"
    if "timed out" in low or "timeout" in low:
        return "timeout"
    if "403" in low:
        return "http_403"
    if "404" in low:
        return "http_404"
    if "429" in low:
        return "http_429"
    if re.search(r"\b5\d{2}\b", low) and "http" in low:
        return "http_5xx"
    if "parse error" in low:
        return "parse_error"
    return "other"


def request_with_retry(url, headers):
    last_error = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)  # type: ignore
            if resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                if attempt < REQUEST_RETRIES:
                    time.sleep(0.8 * attempt)
                    continue
            resp.raise_for_status()
            return resp, None
        except Exception as e:
            last_error = str(e)
            if attempt < REQUEST_RETRIES:
                time.sleep(0.8 * attempt)

    return None, last_error or "unknown request error"


def fetch_rss_url(url, source, max_items_per_source):
    resp, err = request_with_retry(url, HEADERS)
    if resp is None:
        return None, err

    items = []
    try:
        soup = BeautifulSoup(resp.content, "xml")  # type: ignore
        entries = soup.find_all("item") or soup.find_all("entry")  # type: ignore

        for entry in entries[:max_items_per_source]:  # type: ignore
            title_tag = entry.find("title")  # type: ignore
            date_tag = (
                entry.find("pubDate")  # type: ignore
                or entry.find("published")  # type: ignore
                or entry.find("updated")  # type: ignore
                or entry.find("dc:date")  # type: ignore
            )
            link_tag = entry.find("link")  # type: ignore
            id_tag = entry.find("id")  # type: ignore
            desc_tag = (
                entry.find("description")  # type: ignore
                or entry.find("summary")  # type: ignore
                or entry.find("content")  # type: ignore
            )

            title_text = normalize_text(title_tag.get_text(" ", strip=True) if title_tag else "")
            raw_date = normalize_text(date_tag.get_text(strip=True) if date_tag else "")
            desc_text = strip_html(desc_tag.get_text(" ", strip=True) if desc_tag else "")

            link_text = ""
            if link_tag:
                link_text = (
                    link_tag.get("href")  # type: ignore
                    or link_tag.get_text(strip=True)
                    or ""
                )
            if not link_text and id_tag:
                link_text = normalize_text(id_tag.get_text(strip=True))

            date_iso, published_at, parsed_ok = parse_feed_datetime(raw_date)
            canonical_url = canonicalize_url(link_text)
            domain = urlsplit(canonical_url).netloc.lower() if canonical_url else ""

            if not title_text:
                continue

            items.append({
                "source": source["name"],  # type: ignore
                "source_type": "rss",
                "category": source["category"],  # type: ignore
                "weight": source.get("weight", 1),  # type: ignore
                "title": title_text,
                "date": date_iso or "",
                "published_at": published_at,
                "raw_date": raw_date,
                "date_parsed_ok": parsed_ok,
                "url": canonical_url,  # type: ignore
                "domain": domain,
                "description": desc_text[:500],  # type: ignore
            })
    except Exception as e:
        return None, f"parse error: {e}"

    return items, None


def fetch_rss(source, max_items_per_source):
    urls_to_try = [source["url"]]  # type: ignore
    if source.get("fallback_url"):  # type: ignore
        urls_to_try.append(source["fallback_url"])  # type: ignore

    last_error = None
    attempted_urls = []
    for idx, attempt_url in enumerate(urls_to_try):
        attempted_urls.append(attempt_url)
        items, err = fetch_rss_url(attempt_url, source, max_items_per_source)
        if items is None:
            last_error = err
            continue

        # 主地址空数据且存在 fallback 时，尝试 fallback
        if not items and idx < len(urls_to_try) - 1:
            continue
        return {
            "items": items,
            "error": None,
            "fetched_url": attempt_url,
            "attempted_urls": attempted_urls,
            "used_fallback": idx > 0,
        }

    return {
        "items": None,
        "error": last_error or "rss fetch failed",
        "fetched_url": urls_to_try[-1],  # type: ignore
        "attempted_urls": attempted_urls,
        "used_fallback": False,
    }


def fetch_reddit_json(source, max_items_per_source):
    resp, err = request_with_retry(source["url"], REDDIT_HEADERS)  # type: ignore
    if resp is None:
        return {
            "items": None,
            "error": err,
            "fetched_url": source["url"],  # type: ignore
            "attempted_urls": [source["url"]],  # type: ignore
            "used_fallback": False,
        }

    items = []
    try:
        posts = resp.json().get("data", {}).get("children", [])  # type: ignore
        for post in posts[:max_items_per_source]:  # type: ignore
            p = post.get("data", {})  # type: ignore
            title = normalize_text(p.get("title", ""))  # type: ignore
            if not title:
                continue

            created = p.get("created_utc")  # type: ignore
            date_iso = None
            published_at = None
            parsed_ok = False
            if created:
                try:
                    dt = datetime.fromtimestamp(created, tz=timezone.utc)  # type: ignore
                    date_iso = dt.date().isoformat()  # type: ignore
                    published_at = dt.isoformat()  # type: ignore
                    parsed_ok = True
                except Exception:
                    date_iso = None
                    published_at = None

            url = canonicalize_url(p.get("url", ""))  # type: ignore
            domain = urlsplit(url).netloc.lower() if url else ""

            items.append({
                "source": source["name"],  # type: ignore
                "source_type": "reddit",
                "category": source["category"],  # type: ignore
                "weight": source.get("weight", 1),  # type: ignore
                "title": title,
                "date": date_iso or "",
                "published_at": published_at,
                "raw_date": str(created or ""),
                "date_parsed_ok": parsed_ok,
                "url": url,
                "domain": domain,
                "description": normalize_text((p.get("selftext") or "")[:500]),  # type: ignore
            })
    except Exception as e:
        return {
            "items": None,
            "error": f"json parse error: {e}",
            "fetched_url": source["url"],  # type: ignore
            "attempted_urls": [source["url"]],  # type: ignore
            "used_fallback": False,
        }

    return {
        "items": items,
        "error": None,
        "fetched_url": source["url"],  # type: ignore
        "attempted_urls": [source["url"]],  # type: ignore
        "used_fallback": False,
    }


def fetch_source(source, max_items_per_source):
    if source.get("type") == "reddit_json":  # type: ignore
        return fetch_reddit_json(source, max_items_per_source)
    return fetch_rss(source, max_items_per_source)


def deduplicate_within_source(items):
    """只做来源内去重，保留跨来源同题以便后续交叉佐证"""
    seen_url = set()
    seen_title_date = set()
    unique = []
    rejected = []

    for item in items:
        source = item.get("source", "")  # type: ignore
        title_key = normalize_text(item.get("title", "")).lower()  # type: ignore
        date_key = item.get("date", "")  # type: ignore
        url_key = item.get("url", "")  # type: ignore

        if url_key:
            k = (source, url_key)
            if k in seen_url:
                rejected.append({
                    "reason": "duplicate_within_source_url",
                    "source": source,
                    "title": item.get("title", "")[:160],  # type: ignore
                    "url": url_key,
                })
                continue
            seen_url.add(k)

        kd = (source, title_key, date_key)
        if kd in seen_title_date:
            rejected.append({
                "reason": "duplicate_within_source_title_date",
                "source": source,
                "title": item.get("title", "")[:160],  # type: ignore
                "url": url_key,
            })
            continue
        seen_title_date.add(kd)
        unique.append(item)

    return unique, rejected


def evaluate_base_authenticity(items, lookback_days, policy):
    """基础评估：相关性 + 日期有效性 + 来源质量 + 文本质量"""
    today = datetime.now(timezone.utc).date()  # type: ignore
    oldest = today - timedelta(days=lookback_days)  # type: ignore
    accepted_for_corroboration = []
    rejected = []

    for item in items:
        title = normalize_text(item.get("title", ""))  # type: ignore
        description = strip_html(item.get("description", ""))  # type: ignore
        text = f"{title} {description}".lower()

        hard_reasons = []
        flags = []
        score = 0

        if not title:
            hard_reasons.append("empty_title")

        ufo_hits = keyword_hits(text, UFO_KEYWORDS)
        crisis_hits = keyword_hits(text, CRISIS_KEYWORDS)
        hard_crisis_hits = keyword_hits(text, CRISIS_HARD_SIGNAL_KEYWORDS)
        official_action_hits = keyword_hits(text, OFFICIAL_ACTION_KEYWORDS)
        title_ufo_hits = keyword_hits(title.lower(), UFO_KEYWORDS)
        title_crisis_hits = keyword_hits(title.lower(), CRISIS_KEYWORDS)
        title_hard_crisis_hits = keyword_hits(title.lower(), CRISIS_HARD_SIGNAL_KEYWORDS)
        title_official_action_hits = keyword_hits(title.lower(), OFFICIAL_ACTION_KEYWORDS)
        title_actor_hits = keyword_hits(title.lower(), TITLE_NATIONAL_POLITICAL_KEYWORDS)
        if not ufo_hits and not crisis_hits and not hard_crisis_hits and not official_action_hits:
            hard_reasons.append("no_relevance_keywords")

        initial_category = item.get("category", "ufo")  # type: ignore
        crisis_signal_strength = len(crisis_hits) + len(hard_crisis_hits) + len(official_action_hits)
        if len(ufo_hits) > crisis_signal_strength:
            final_category = "ufo"
        elif crisis_signal_strength > len(ufo_hits):
            final_category = "crisis"
        else:
            final_category = initial_category
        item["category"] = final_category  # type: ignore
        item["ufo_relevance"] = len(ufo_hits)  # type: ignore
        item["crisis_relevance"] = len(crisis_hits)  # type: ignore
        item["ufo_keywords"] = ufo_hits[:12]  # type: ignore
        item["crisis_keywords"] = crisis_hits[:12]  # type: ignore
        item["hard_crisis_keywords"] = hard_crisis_hits[:12]  # type: ignore
        item["official_action_keywords"] = official_action_hits[:12]  # type: ignore
        item["title_ufo_keywords"] = title_ufo_hits[:12]  # type: ignore
        item["title_crisis_keywords"] = title_crisis_hits[:12]  # type: ignore
        item["title_hard_crisis_keywords"] = title_hard_crisis_hits[:12]  # type: ignore
        item["title_official_action_keywords"] = title_official_action_hits[:12]  # type: ignore
        item["title_national_actor_keywords"] = title_actor_hits[:12]  # type: ignore
        political_hits = keyword_hits(text, POLITICAL_CONTEXT_KEYWORDS)
        item["political_context_relevance"] = len(political_hits)  # type: ignore
        item["political_context_keywords"] = political_hits[:12]  # type: ignore
        national_hits = keyword_hits(text, NATIONAL_POLITICAL_CONTEXT_KEYWORDS)
        item["national_context_relevance"] = len(national_hits)  # type: ignore
        item["national_context_keywords"] = national_hits[:12]  # type: ignore

        weight = item.get("weight", 1)  # type: ignore
        if weight >= 3:
            score += 40
        elif weight == 2:
            score += 30
        else:
            score += 15

        if item.get("source_type") == "rss":  # type: ignore
            score += 15
        elif item.get("source_type") == "reddit":  # type: ignore
            score += 4
            flags.append("community_source")

        is_aggregator = source_is_aggregator(item.get("source", ""))  # type: ignore
        if is_aggregator:
            score -= 12
            flags.append("aggregator_source")

        url = item.get("url", "")  # type: ignore
        if url.startswith("https://"):
            score += 5
        elif url.startswith("http://"):
            score -= 4
            flags.append("non_https_url")
        else:
            score -= 8
            flags.append("missing_or_invalid_url")

        if item.get("domain"):  # type: ignore
            score += 3
        else:
            score -= 6
            flags.append("missing_domain")

        if 25 <= len(title) <= 180:
            score += 6
        else:
            score -= 6
            flags.append("abnormal_title_length")

        if looks_spammy_title(title):
            score -= 10
            flags.append("spammy_title_pattern")

        if policy["enforce_opinion_filter"] and looks_opinion_or_commentary(title):  # type: ignore
            hard_reasons.append("opinion_or_commentary")
        if policy["enforce_roundup_filter"] and looks_roundup_or_mixed_headline(title):  # type: ignore
            hard_reasons.append("roundup_or_mixed_headline")

        raw_date = item.get("date")  # type: ignore
        date_obj = parse_iso_date(raw_date)
        if item.get("date_parsed_ok"):  # type: ignore
            score += 8
        else:
            score -= 12
            hard_reasons.append("date_parse_failed")

        if date_obj is None:
            hard_reasons.append("invalid_date")
        else:
            if date_obj > today + timedelta(days=1):  # type: ignore
                hard_reasons.append("future_dated_item")
            elif date_obj < oldest:
                hard_reasons.append("stale_item_outside_lookback")
            else:
                score += 6

        score += min(len(ufo_hits) + len(crisis_hits), 8)
        score += min(len(official_action_hits), 4)

        if final_category == "crisis":
            if political_hits:
                score += min(len(political_hits), 4)
            else:
                hard_reasons.append("crisis_without_political_context")

            if policy.get("require_crisis_national_context", True) and not national_hits:  # type: ignore
                hard_reasons.append("crisis_without_national_context")
            if policy.get("require_crisis_hard_signal", True) and not hard_crisis_hits and not official_action_hits:  # type: ignore
                hard_reasons.append("crisis_without_hard_signal")
            if policy["enforce_crisis_title_actor"] and not title_actor_hits:  # type: ignore
                hard_reasons.append("crisis_title_without_national_actor")
            if not title_crisis_hits and not title_hard_crisis_hits and not title_official_action_hits:
                hard_reasons.append("crisis_signal_only_in_description")
            if (
                policy["enforce_crisis_title_hard_signal"]  # type: ignore
                and not title_hard_crisis_hits
                and not title_official_action_hits
            ):
                hard_reasons.append("crisis_hard_signal_only_in_description")

        if final_category == "ufo":
            if policy["enforce_ufo_title_signal"] and not title_ufo_hits:  # type: ignore
                hard_reasons.append("ufo_signal_only_in_description")

        item["authenticity"] = {  # type: ignore
            "base_score": max(0, score),
            "final_score": max(0, score),
            "label": confidence_label(max(0, score)),
            "flags": flags,
            "is_aggregator": is_aggregator,
            "corroboration_count": 0,
            "trusted_corroboration": 0,
        }

        if hard_reasons:
            rejected.append({
                "reason": ",".join(sorted(set(hard_reasons))),
                "source": item.get("source"),  # type: ignore
                "title": title[:180],  # type: ignore
                "date": item.get("date", ""),  # type: ignore
                "url": url,
            })
            continue

        accepted_for_corroboration.append(item)

    return accepted_for_corroboration, rejected


def apply_corroboration(items):
    clusters = defaultdict(list)
    for item in items:
        fp, tokens = claim_fingerprint(item)
        item["claim_fingerprint"] = fp  # type: ignore
        item["claim_tokens"] = tokens  # type: ignore
        clusters[(item.get("category"), fp)].append(item)  # type: ignore

    for group in clusters.values():  # type: ignore
        source_set = {i.get("source") for i in group if i.get("source")}  # type: ignore
        trusted_sources = {i.get("source") for i in group if source_is_trusted(i)}  # type: ignore
        corroboration_count = len(source_set)
        trusted_count = len(trusted_sources)

        bonus = 0
        if trusted_count >= 3:
            bonus += 22
        elif trusted_count == 2:
            bonus += 13
        elif trusted_count == 1:
            bonus += 5

        if corroboration_count >= 5:
            bonus += 5
        elif corroboration_count >= 3:
            bonus += 2

        for item in group:
            base = item["authenticity"]["base_score"]  # type: ignore
            final = min(100, base + bonus)
            item["authenticity"]["final_score"] = final  # type: ignore
            item["authenticity"]["label"] = confidence_label(final)  # type: ignore
            item["authenticity"]["corroboration_count"] = corroboration_count  # type: ignore
            item["authenticity"]["trusted_corroboration"] = trusted_count  # type: ignore


def filter_by_policy(items, policy):
    accepted = []
    rejected = []
    min_score = policy["min_score"]  # type: ignore

    for item in items:
        auth = item.get("authenticity", {})  # type: ignore
        reasons = []
        final_score = auth.get("final_score", 0)  # type: ignore
        trusted_c = auth.get("trusted_corroboration", 0)  # type: ignore
        is_aggregator = auth.get("is_aggregator", False)  # type: ignore
        source_type = item.get("source_type")  # type: ignore
        weight = item.get("weight", 1)  # type: ignore

        if final_score < min_score:
            reasons.append("score_below_threshold")

        if policy["strict_source_rules"]:  # type: ignore
            if weight == 1 and trusted_c < 2:
                reasons.append("low_weight_without_trusted_corroboration")

            if source_type == "reddit" and trusted_c < 2:
                reasons.append("community_source_without_trusted_corroboration")

            if is_aggregator and trusted_c < 2:
                reasons.append("aggregator_without_trusted_corroboration")

            if weight == 2 and trusted_c == 0 and final_score < 78:
                reasons.append("tier2_source_needs_more_evidence")

        if reasons:
            rejected.append({
                "reason": ",".join(sorted(set(reasons))),
                "source": item.get("source"),  # type: ignore
                "title": item.get("title", "")[:180],  # type: ignore
                "date": item.get("date", ""),  # type: ignore
                "url": item.get("url", ""),  # type: ignore
                "score": final_score,
            })
            continue

        accepted.append(item)

    return accepted, rejected


def build_corroboration_timeline(group):
    timeline = []
    seen = set()
    for item in group:
        source = str(item.get("source", "") or "")
        date = str(item.get("date", "") or "")
        published_at = str(item.get("published_at", "") or "")
        url = str(item.get("url", "") or "")
        if not source and not date:
            continue
        k = (source, date, published_at, url)
        if k in seen:
            continue
        seen.add(k)
        timeline.append({
            "source": source,
            "date": date,
            "published_at": published_at,
            "source_type": str(item.get("source_type", "") or ""),
            "weight": int(item.get("weight", 1) or 1),
            "url": url,
            "is_official_source": source_is_official(source),
        })

    timeline.sort(
        key=lambda x: (
            x.get("published_at", "") == "",
            x.get("published_at", ""),  # type: ignore
            x.get("date", "") == "",
            x.get("date", ""),  # type: ignore
            -int(x.get("weight", 1) or 1),  # type: ignore
            x.get("source", ""),  # type: ignore
        )
    )
    return timeline[:24]  # type: ignore


def enrich_official_media_timeline_metrics(event):
    timeline = event.get("corroboration_timeline", []) or []  # type: ignore
    official_dates = []
    media_dates_rss = []
    media_dates_any = []
    official_dt = []
    media_dt_rss = []
    media_dt_any = []
    for row in timeline:
        d = parse_iso_date(row.get("date"))  # type: ignore
        dt = parse_iso_datetime(row.get("published_at"))  # type: ignore
        if d is None:
            continue
        if bool(row.get("is_official_source")):  # type: ignore
            official_dates.append(d)
            if dt is not None:
                official_dt.append(dt)
            continue
        media_dates_any.append(d)  # type: ignore
        if dt is not None:
            media_dt_any.append(dt)
        if row.get("source_type") == "rss":  # type: ignore
            media_dates_rss.append(d)
            if dt is not None:
                media_dt_rss.append(dt)

    media_dates = media_dates_rss if media_dates_rss else media_dates_any
    media_dates_dt = media_dt_rss if media_dt_rss else media_dt_any
    first_official = min(official_dates).isoformat() if official_dates else None  # type: ignore
    first_media = min(media_dates).isoformat() if media_dates else None  # type: ignore
    first_official_dt = min(official_dt).isoformat() if official_dt else None  # type: ignore
    first_media_dt = min(media_dates_dt).isoformat() if media_dates_dt else None  # type: ignore

    lag_days = None
    lag_hours = None
    official_leads = None
    official_leads_ts = None
    if first_official and first_media:
        lag_days = (parse_iso_date(first_media) - parse_iso_date(first_official)).days  # type: ignore
        official_leads = lag_days >= 0
    if first_official_dt and first_media_dt:
        lag_hours = round((parse_iso_datetime(first_media_dt) - parse_iso_datetime(first_official_dt)).total_seconds() / 3600.0, 3)  # type: ignore
        official_leads_ts = lag_hours >= 0
    lag_basis = "timestamp" if lag_hours is not None else ("date" if lag_days is not None else "none")

    event["first_official_date"] = first_official  # type: ignore
    event["first_media_date"] = first_media  # type: ignore
    event["first_official_published_at"] = first_official_dt  # type: ignore
    event["first_media_published_at"] = first_media_dt  # type: ignore
    event["official_to_media_lag_days"] = lag_days  # type: ignore
    event["official_to_media_lag_hours"] = lag_hours  # type: ignore
    event["official_leads_media"] = official_leads  # type: ignore
    event["official_leads_media_by_timestamp"] = official_leads_ts  # type: ignore
    event["official_media_lag_basis"] = lag_basis  # type: ignore
    event["official_timeline_observations"] = len(official_dates)  # type: ignore
    event["media_timeline_observations"] = len(media_dates)  # type: ignore
    event["official_timeline_timestamp_observations"] = len(official_dt)  # type: ignore
    event["media_timeline_timestamp_observations"] = len(media_dates_dt)  # type: ignore
    return event


def collapse_claim_clusters(items):
    """将通过过滤的同一事件多来源条目合并为事件级记录，避免后续关联重复爆炸。"""
    groups = defaultdict(list)
    for item in items:
        fp = item.get("claim_fingerprint")  # type: ignore
        if not fp:
            fp, tokens = claim_fingerprint(item)
            item["claim_fingerprint"] = fp  # type: ignore
            item["claim_tokens"] = tokens  # type: ignore
        groups[(item.get("category"), fp)].append(item)  # type: ignore

    collapsed = []
    for (category, fp), group in groups.items():  # type: ignore
        ranked = sorted(
            group,
            key=lambda x: (
                x.get("authenticity", {}).get("final_score", 0),  # type: ignore
                x.get("weight", 1),  # type: ignore
                1 if x.get("source_type") == "rss" else 0,  # type: ignore
                len(x.get("title", "")),  # type: ignore
            ),
            reverse=True,
        )
        primary = dict(ranked[0])  # type: ignore

        sources = sorted({g.get("source") for g in group if g.get("source")})  # type: ignore
        domains = sorted({g.get("domain") for g in group if g.get("domain")})  # type: ignore
        urls = []
        seen_urls = set()
        for g in ranked:
            u = g.get("url", "")  # type: ignore
            if u and u not in seen_urls:
                seen_urls.add(u)
                urls.append(u)

        trusted_sources = sorted({g.get("source") for g in group if source_is_trusted(g)})  # type: ignore

        primary["category"] = category  # type: ignore
        primary["claim_fingerprint"] = fp  # type: ignore
        primary["cluster_size"] = len(group)  # type: ignore
        primary["corroborated_sources"] = sources  # type: ignore
        primary["corroborated_domains"] = domains  # type: ignore
        primary["evidence_urls"] = urls[:12]  # type: ignore
        primary["primary_source"] = ranked[0].get("source")  # type: ignore
        primary["corroboration_timeline"] = build_corroboration_timeline(group)  # type: ignore
        if len(sources) > 1:
            primary["source_type"] = "multi"  # type: ignore

        auth = primary.setdefault("authenticity", {})  # type: ignore
        auth["corroboration_count"] = max(auth.get("corroboration_count", 0), len(sources))  # type: ignore
        auth["trusted_corroboration"] = max(auth.get("trusted_corroboration", 0), len(trusted_sources))  # type: ignore

        collapsed.append(enrich_official_media_timeline_metrics(primary))

    return merge_crisis_events_by_signature(collapsed)


def _pick_best_anchor(text, candidates, default="na"):
    hits = keyword_hits(text, candidates)
    if not hits:
        return default
    # Prefer longer phrase anchors for more stable event signatures.
    return sorted(set(hits), key=len, reverse=True)[0]


def crisis_event_signature(item):
    """
    Build a coarse event signature for crisis items to reduce same-day media burst duplication.
    Signature = date + action anchor + topic anchor + actor anchor.
    """
    date_key = item.get("date", "") or "na"  # type: ignore
    title = normalize_text(item.get("title", "")).lower()  # type: ignore
    desc = strip_html(item.get("description", "")).lower()  # type: ignore
    text = f"{title} {desc}"

    action_anchor = _pick_best_anchor(text, OFFICIAL_ACTION_KEYWORDS + CRISIS_HARD_SIGNAL_KEYWORDS, default="na")
    topic_anchor = _pick_best_anchor(text, CRISIS_TOPIC_KEYWORDS, default="na")
    actor_anchor = _pick_best_anchor(
        title,
        [
            "trump", "biden", "white house", "congress", "senate", "house",
            "supreme court", "doj", "department of justice", "fbi", "pentagon",
        ],
        default="na",
    )
    return f"{date_key}|act:{action_anchor}|topic:{topic_anchor}|actor:{actor_anchor}"


def merge_crisis_events_by_signature(events):
    crisis = [e for e in events if e.get("category") == "crisis"]  # type: ignore
    others = [e for e in events if e.get("category") != "crisis"]  # type: ignore
    if not crisis:
        return events

    buckets = defaultdict(list)
    for row in crisis:
        buckets[crisis_event_signature(row)].append(row)  # type: ignore

    merged = []
    for _, group in buckets.items():  # type: ignore
        ranked = sorted(
            group,
            key=lambda x: (
                x.get("authenticity", {}).get("final_score", 0),  # type: ignore
                x.get("cluster_size", 1),  # type: ignore
                len(x.get("title", "")),  # type: ignore
            ),
            reverse=True,
        )
        base = dict(ranked[0])  # type: ignore

        merged_sources = []
        merged_domains = []
        merged_urls = []
        merged_timeline = []
        seen_s = set()
        seen_d = set()
        seen_u = set()
        seen_t = set()
        total_cluster = 0
        for r in group:
            total_cluster += int(r.get("cluster_size", 1) or 1)  # type: ignore
            for s in r.get("corroborated_sources", []) or []:  # type: ignore
                if s and s not in seen_s:
                    seen_s.add(s)
                    merged_sources.append(s)
            for d in r.get("corroborated_domains", []) or []:  # type: ignore
                if d and d not in seen_d:
                    seen_d.add(d)
                    merged_domains.append(d)
            for u in r.get("evidence_urls", []) or []:  # type: ignore
                if u and u not in seen_u:
                    seen_u.add(u)
                    merged_urls.append(u)
            for t in r.get("corroboration_timeline", []) or []:  # type: ignore
                k = (
                    t.get("source", ""),  # type: ignore
                    t.get("date", ""),  # type: ignore
                    t.get("url", ""),  # type: ignore
                )
                if k in seen_t:
                    continue
                seen_t.add(k)
                merged_timeline.append(t)

        base["cluster_size"] = total_cluster  # type: ignore
        base["corroborated_sources"] = merged_sources  # type: ignore
        base["corroborated_domains"] = merged_domains  # type: ignore
        base["evidence_urls"] = merged_urls[:16]  # type: ignore
        if merged_timeline:
            merged_timeline.sort(
                key=lambda x: (
                    x.get("date", "") == "",
                    x.get("date", ""),  # type: ignore
                    -int(x.get("weight", 1) or 1),  # type: ignore
                    x.get("source", ""),  # type: ignore
                )
            )
            base["corroboration_timeline"] = merged_timeline[:24]  # type: ignore
        base["merged_claims"] = len(group)  # type: ignore
        auth = base.setdefault("authenticity", {})  # type: ignore
        auth["corroboration_count"] = max(auth.get("corroboration_count", 0), len(merged_sources))  # type: ignore
        merged.append(enrich_official_media_timeline_metrics(base))

    # Keep deterministic ordering for downstream panel snapshots.
    out = others + merged
    out.sort(
        key=lambda x: (
            x.get("date", ""),  # type: ignore
            x.get("category", ""),  # type: ignore
            x.get("authenticity", {}).get("final_score", 0),  # type: ignore
            x.get("title", ""),  # type: ignore
        ),
        reverse=True,
    )
    return out


def split_and_sort_news(items):
    def sort_key(x):
        return (
            x.get("date", ""),  # type: ignore
            x.get("authenticity", {}).get("final_score", 0),  # type: ignore
            x.get("weight", 1),  # type: ignore
        )

    ufo_news = sorted(
        [i for i in items if i.get("category") == "ufo"],  # type: ignore
        key=sort_key,
        reverse=True,
    )
    crisis_news = sorted(
        [i for i in items if i.get("category") == "crisis"],  # type: ignore
        key=sort_key,
        reverse=True,
    )
    return ufo_news, crisis_news


def classify_official_lead_reason(event):
    off_obs = int(event.get("official_timeline_observations", 0) or 0)  # type: ignore
    med_obs = int(event.get("media_timeline_observations", 0) or 0)  # type: ignore
    lag_days = event.get("official_to_media_lag_days")  # type: ignore
    lag_hours = event.get("official_to_media_lag_hours")  # type: ignore

    if off_obs <= 0:
        return "no_official_source_in_timeline"
    if med_obs <= 0:
        return "no_media_source_in_timeline"
    if lag_days is None:
        return "missing_lag_day_signal"
    if isinstance(lag_hours, (int, float)) and lag_hours < 0:
        return "media_leads_by_timestamp"
    if lag_days < 0:
        return "media_leads_by_day"
    if lag_days == 0 and lag_hours is None:
        return "same_day_without_timestamp"
    if lag_days == 0 and isinstance(lag_hours, (int, float)) and lag_hours == 0:
        return "same_timestamp_tie"
    if lag_days == 0:
        return "official_leads_same_day_by_timestamp"
    return "official_leads_cross_day"


def build_official_lead_diagnostics(ufo_news):
    rows = []
    reason_counter = Counter()
    for e in ufo_news:
        reason = classify_official_lead_reason(e)
        reason_counter[reason] += 1  # type: ignore
        lag_days = e.get("official_to_media_lag_days")  # type: ignore
        lag_hours = e.get("official_to_media_lag_hours")  # type: ignore
        lead_nonnegative_day = isinstance(lag_days, (int, float)) and lag_days >= 0
        lead_strict = (
            (isinstance(lag_days, (int, float)) and lag_days > 0)
            or (lag_days == 0 and isinstance(lag_hours, (int, float)) and lag_hours > 0)
        )
        rows.append({
            "claim_fingerprint": e.get("claim_fingerprint"),  # type: ignore
            "date": e.get("date"),  # type: ignore
            "title": e.get("title"),  # type: ignore
            "cluster_size": int(e.get("cluster_size", 1) or 1),  # type: ignore
            "primary_source": e.get("primary_source"),  # type: ignore
            "first_official_date": e.get("first_official_date"),  # type: ignore
            "first_media_date": e.get("first_media_date"),  # type: ignore
            "first_official_published_at": e.get("first_official_published_at"),  # type: ignore
            "first_media_published_at": e.get("first_media_published_at"),  # type: ignore
            "official_to_media_lag_days": lag_days,
            "official_to_media_lag_hours": lag_hours,
            "official_media_lag_basis": e.get("official_media_lag_basis"),  # type: ignore
            "official_timeline_observations": int(e.get("official_timeline_observations", 0) or 0),  # type: ignore
            "media_timeline_observations": int(e.get("media_timeline_observations", 0) or 0),  # type: ignore
            "official_timeline_timestamp_observations": int(e.get("official_timeline_timestamp_observations", 0) or 0),  # type: ignore
            "media_timeline_timestamp_observations": int(e.get("media_timeline_timestamp_observations", 0) or 0),  # type: ignore
            "lead_nonnegative_day": bool(lead_nonnegative_day),
            "lead_strict_candidate": bool(lead_strict),
            "diagnostic_reason": reason,
        })

    rows.sort(
        key=lambda x: (
            0 if x.get("lead_strict_candidate") else 1,  # type: ignore
            0 if x.get("lead_nonnegative_day") else 1,  # type: ignore
            -(x.get("official_to_media_lag_days") or -999),  # type: ignore
            -(x.get("cluster_size") or 1),  # type: ignore
            x.get("date", "") or "",  # type: ignore
        )
    )

    total = len(rows)
    with_official = sum(1 for r in rows if int(r.get("official_timeline_observations", 0) or 0) > 0)
    with_media = sum(1 for r in rows if int(r.get("media_timeline_observations", 0) or 0) > 0)
    with_lag_day = sum(1 for r in rows if isinstance(r.get("official_to_media_lag_days"), (int, float)))
    with_lag_hour = sum(1 for r in rows if isinstance(r.get("official_to_media_lag_hours"), (int, float)))
    lead_nonnegative_day = sum(1 for r in rows if bool(r.get("lead_nonnegative_day")))
    lead_strict = sum(1 for r in rows if bool(r.get("lead_strict_candidate")))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # type: ignore
        "summary": {
            "total_ufo_events": total,
            "with_official_source": with_official,
            "with_media_source": with_media,
            "with_lag_days": with_lag_day,
            "with_lag_hours": with_lag_hour,
            "official_leads_nonnegative_day_events": lead_nonnegative_day,
            "official_lead_strict_candidates": lead_strict,
            "top_blockers": [
                {"reason": reason, "count": count}
                for reason, count in reason_counter.most_common(10)
            ],
        },
        "events": rows[:200],
    }


def print_source_summary(sources, counts, errors):
    t = Table(title="来源抓取摘要", box=box.SIMPLE, show_lines=False)  # type: ignore
    t.add_column("来源名称", width=32)
    t.add_column("类型", width=8, justify="center")
    t.add_column("权重", width=5, justify="center")
    t.add_column("条数", width=6, justify="right")
    t.add_column("状态", width=5, justify="center")
    wc = {3: "green", 2: "yellow", 1: "dim"}

    for src in sources:
        name = src["name"]  # type: ignore
        count = counts.get(name)  # type: ignore
        weight = src.get("weight", 1)  # type: ignore
        if count is None:
            status = "[red]✗[/red]"
            count_str = "[red]失败[/red]"  # type: ignore
        else:
            status = "[green]✓[/green]"
            count_str = str(count)

        t.add_row(
            name[:30],  # type: ignore
            src.get("type", "rss").upper(),  # type: ignore
            f"[{wc.get(weight, 'white')}]{weight}[/{wc.get(weight, 'white')}]",  # type: ignore
            count_str,
            status,
        )

    console.print(t)  # type: ignore

    failed = [(k, v) for k, v in errors.items() if v]  # type: ignore
    if failed:
        console.print("[dim]失败来源（摘要）：[/dim]")  # type: ignore
        for name, err in failed[:8]:  # type: ignore
            console.print(f"[dim]- {name}: {err[:140]}[/dim]")  # type: ignore


def print_quality_summary(raw_count, unique_count, candidate_count, accepted_count, rejected_items):
    t = Table(title="真实性过滤摘要", box=box.SIMPLE)  # type: ignore
    t.add_column("阶段", width=24)
    t.add_column("条数", justify="right", width=10)
    t.add_row("抓取原始条目", str(raw_count))
    t.add_row("来源内去重后", str(unique_count))
    t.add_row("通过基础有效性检查", str(candidate_count))
    t.add_row("最终通过真实性过滤", f"[green]{accepted_count}[/green]")
    t.add_row("被拒绝条目", f"[red]{len(rejected_items)}[/red]")
    console.print(t)  # type: ignore

    if rejected_items:
        counter = Counter()
        for i in rejected_items:
            for reason in (i.get("reason", "") or "").split(","):  # type: ignore
                if reason:
                    counter[reason] += 1  # type: ignore

        top = counter.most_common(8)
        if top:
            rt = Table(title="主要拒绝原因", box=box.SIMPLE)  # type: ignore
            rt.add_column("原因", width=52)
            rt.add_column("次数", justify="right", width=10)
            for reason, n in top:
                rt.add_row(reason, str(n))
            console.print(rt)  # type: ignore


def build_source_health_report(source_meta, all_sources, active_sources, source_stats):
    total = len(active_sources)
    available = sum(1 for r in source_stats if r.get("status") != "failed")  # type: ignore
    success_with_items = sum(1 for r in source_stats if r.get("status") == "ok")  # type: ignore
    failed = [r for r in source_stats if r.get("status") == "failed"]  # type: ignore
    fallback_used = sum(1 for r in source_stats if r.get("used_fallback"))  # type: ignore

    categories = sorted({s.get("category", "unknown") for s in active_sources})  # type: ignore
    by_category = {}
    for cat in categories:
        rows = [r for r in source_stats if r.get("category") == cat]  # type: ignore
        trusted_rows = [r for r in rows if r.get("type") == "rss" and int(r.get("weight", 1)) >= 2]  # type: ignore
        by_category[cat] = {  # type: ignore
            "active": len(rows),
            "available": sum(1 for r in rows if r.get("status") != "failed"),  # type: ignore
            "success_with_items": sum(1 for r in rows if r.get("status") == "ok"),  # type: ignore
            "failed": sum(1 for r in rows if r.get("status") == "failed"),  # type: ignore
            "trusted_active": len(trusted_rows),
            "trusted_success_with_items": sum(1 for r in trusted_rows if r.get("status") == "ok"),  # type: ignore
        }

    inactive_by_category = defaultdict(list)
    for src in all_sources:
        if src.get("active", True):  # type: ignore
            continue
        inactive_by_category[src.get("category", "unknown")].append(src)  # type: ignore
    for cat in inactive_by_category:
        inactive_by_category[cat].sort(key=lambda x: int(x.get("weight", 1)), reverse=True)  # type: ignore

    recommendations = []
    for cat, stats in by_category.items():  # type: ignore
        if stats["trusted_success_with_items"] < 2:  # type: ignore
            candidates = []
            for src in inactive_by_category.get(cat, [])[:3]:  # type: ignore
                candidates.append({
                    "name": src.get("name"),  # type: ignore
                    "weight": int(src.get("weight", 1)),  # type: ignore
                    "type": src.get("type", "rss"),  # type: ignore
                    "url": src.get("url", ""),  # type: ignore
                })
            recommendations.append({
                "category": cat,
                "issue": "trusted_source_coverage_low",
                "detail": (
                    f"trusted_success_with_items={stats['trusted_success_with_items']} (<2), "  # type: ignore
                    "建议启用备用来源或修复失败源"
                ),
                "backup_candidates": candidates,
            })

    for row in failed[:8]:  # type: ignore
        if row.get("type") == "rss" and not row.get("fallback_url"):  # type: ignore
            recommendations.append({
                "category": row.get("category"),  # type: ignore
                "issue": "missing_fallback_url",
                "detail": f"{row.get('source')} 抓取失败且未配置 fallback_url",  # type: ignore
                "backup_candidates": [],
            })

    error_type_counter = Counter(classify_source_error(r.get("error")) for r in failed)  # type: ignore

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # type: ignore
        "source_config_last_updated": source_meta.get("last_updated"),  # type: ignore
        "total_active_sources": total,
        "available_sources": available,
        "sources_with_items": success_with_items,
        "failed_sources": len(failed),
        "availability_rate": round((available / total), 4) if total else 0.0,  # type: ignore
        "success_rate_with_items": round((success_with_items / total), 4) if total else 0.0,  # type: ignore
        "fallback_used_count": fallback_used,
        "failed_error_types": dict(error_type_counter),
        "category_coverage": by_category,
        "recommendations": recommendations[:12],  # type: ignore
    }


def print_source_health_summary(source_health):
    t = Table(title="来源健康审计", box=box.SIMPLE)  # type: ignore
    t.add_column("指标", width=28)
    t.add_column("值", justify="right", width=16)
    t.add_row("活跃来源数", str(source_health.get("total_active_sources", 0)))  # type: ignore
    t.add_row("可用来源数", str(source_health.get("available_sources", 0)))  # type: ignore
    t.add_row("有内容来源数", str(source_health.get("sources_with_items", 0)))  # type: ignore
    t.add_row("抓取失败来源数", str(source_health.get("failed_sources", 0)))  # type: ignore
    t.add_row("可用率", f"{source_health.get('availability_rate', 0.0):.2%}")  # type: ignore
    t.add_row("出内容率", f"{source_health.get('success_rate_with_items', 0.0):.2%}")  # type: ignore
    t.add_row("fallback命中数", str(source_health.get("fallback_used_count", 0)))  # type: ignore
    console.print(t)  # type: ignore


def scrape_all(
    policy_name=POLICY_STRICT,
    lookback_days=LOOKBACK_DAYS_DEFAULT,
    max_workers=MAX_WORKERS_DEFAULT,
    max_items_per_source=MAX_ITEMS_PER_SOURCE,
):
    policy_name, policy = resolve_policy(policy_name)
    source_meta, all_sources, sources = load_source_config()
    all_items = []
    counts = {}
    errors = {}
    source_records = {}

    worker_count = max(1, min(max_workers, len(sources)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:  # type: ignore
        futures = {
            pool.submit(fetch_source, src, max_items_per_source): src
            for src in sources
        }

        with Progress(  # type: ignore
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("抓取中...", total=len(futures))  # type: ignore
            for future in as_completed(futures):
                src = futures[future]  # type: ignore
                progress.update(task, description=f"[{src.get('type','rss').upper()}] {src['name'][:30]}")  # type: ignore
                try:
                    result = future.result()  # type: ignore
                except Exception as e:
                    counts[src["name"]] = None  # type: ignore
                    errors[src["name"]] = f"worker exception: {e}"  # type: ignore
                    source_records[src["name"]] = {  # type: ignore
                        "source": src["name"],  # type: ignore
                        "type": src.get("type", "rss"),  # type: ignore
                        "category": src.get("category", "unknown"),  # type: ignore
                        "weight": int(src.get("weight", 1)),  # type: ignore
                        "status": "failed",
                        "item_count": 0,
                        "url": src.get("url", ""),  # type: ignore
                        "fallback_url": src.get("fallback_url"),  # type: ignore
                        "fetched_url": src.get("url", ""),  # type: ignore
                        "used_fallback": False,
                        "attempted_urls": [src.get("url", "")],  # type: ignore
                        "error": f"worker exception: {e}",
                    }
                    progress.advance(task)  # type: ignore
                    continue

                items = result.get("items")  # type: ignore
                err = result.get("error")  # type: ignore
                fetched_url = result.get("fetched_url", src.get("url", ""))  # type: ignore
                attempted_urls = result.get("attempted_urls", [src.get("url", "")])  # type: ignore
                used_fallback = bool(result.get("used_fallback", False))  # type: ignore
                if items is None:
                    counts[src["name"]] = None  # type: ignore
                    errors[src["name"]] = err or "unknown"  # type: ignore
                    source_records[src["name"]] = {  # type: ignore
                        "source": src["name"],  # type: ignore
                        "type": src.get("type", "rss"),  # type: ignore
                        "category": src.get("category", "unknown"),  # type: ignore
                        "weight": int(src.get("weight", 1)),  # type: ignore
                        "status": "failed",
                        "item_count": 0,
                        "url": src.get("url", ""),  # type: ignore
                        "fallback_url": src.get("fallback_url"),  # type: ignore
                        "fetched_url": fetched_url,
                        "used_fallback": used_fallback,
                        "attempted_urls": attempted_urls,
                        "error": err or "unknown",
                    }
                else:  # type: ignore
                    counts[src["name"]] = len(items)  # type: ignore
                    errors[src["name"]] = None  # type: ignore
                    all_items.extend(items)
                    source_records[src["name"]] = {  # type: ignore
                        "source": src["name"],  # type: ignore
                        "type": src.get("type", "rss"),  # type: ignore
                        "category": src.get("category", "unknown"),  # type: ignore
                        "weight": int(src.get("weight", 1)),  # type: ignore
                        "status": "ok" if len(items) > 0 else "empty",  # type: ignore
                        "item_count": len(items),  # type: ignore
                        "url": src.get("url", ""),  # type: ignore
                        "fallback_url": src.get("fallback_url"),  # type: ignore
                        "fetched_url": fetched_url,
                        "used_fallback": used_fallback,
                        "attempted_urls": attempted_urls,
                        "error": None,
                    }

                progress.advance(task)  # type: ignore

    unique_items, dedupe_rejected = deduplicate_within_source(all_items)
    candidates, base_rejected = evaluate_base_authenticity(
        unique_items,
        lookback_days=lookback_days,
        policy=policy,
    )
    apply_corroboration(candidates)
    accepted_items, policy_rejected = filter_by_policy(candidates, policy=policy)
    accepted_events = collapse_claim_clusters(accepted_items)

    rejected_items = dedupe_rejected + base_rejected + policy_rejected
    ufo_news, crisis_news = split_and_sort_news(accepted_events)
    official_lead_diag = build_official_lead_diagnostics(ufo_news)
    source_stats = [source_records.get(src["name"]) for src in sources if source_records.get(src["name"])]  # type: ignore
    source_health = build_source_health_report(source_meta, all_sources, sources, source_stats)
    coverage_audit = build_coverage_audit(
        raw_items=all_items,
        unique_items=unique_items,
        candidates=candidates,
        accepted_events=accepted_events,
        rejected_items=rejected_items,
        source_stats=source_stats,
        lookback_days=lookback_days,
    )

    print_source_summary(sources, counts, errors)
    print_source_health_summary(source_health)
    print_coverage_summary(coverage_audit)
    print_quality_summary(
        raw_count=len(all_items),
        unique_count=len(unique_items),
        candidate_count=len(candidates),
        accepted_count=len(accepted_items),
        rejected_items=rejected_items,
    )

    output = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),  # type: ignore
        "strict_mode": policy_name != POLICY_LENIENT,
        "policy": policy_name,
        "lookback_days": lookback_days,
        "max_items_per_source": max_items_per_source,
        "source_count": len(sources),
        "source_stats": source_stats,
        "source_health": source_health,
        "coverage_audit": coverage_audit,
        "stats": {
            "raw_items": len(all_items),
            "unique_items": len(unique_items),
            "candidates_after_base_checks": len(candidates),
            "accepted_items": len(accepted_items),
            "accepted_events": len(accepted_events),
            "rejected_items": len(rejected_items),
        },
        "events_by_type_count": {
            "ufo": len(ufo_news),
            "crisis": len(crisis_news),
        },
        "authenticity_policy": {
            "strict_min_score": STRICT_MIN_SCORE,
            "strict_balanced_min_score": BALANCED_MIN_SCORE,
            "lenient_min_score": LENIENT_MIN_SCORE,
            "active_policy": policy_name,
            "notes": [
                "严格模式下，低权重来源必须有可信来源佐证",
                "社区与聚合来源必须有至少2个可信来源交叉确认",
                "未来日期、日期解析失败、超出时间窗口的条目会被直接剔除",
            ],
        },
        "official_lead_diagnostics": {
            "summary": official_lead_diag.get("summary", {}),  # type: ignore
            "file": OFFICIAL_LEAD_DIAG_FILE,
        },
        "ufo_news": ufo_news,
        "crisis_news": crisis_news,
        "rejected_news": rejected_items,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SCRAPED_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)  # type: ignore
    with open(OFFICIAL_LEAD_DIAG_FILE, "w", encoding="utf-8") as f:
        json.dump(official_lead_diag, f, ensure_ascii=False, indent=2)  # type: ignore

    console.print(  # type: ignore
        f"\n[green]✓ 抓取完成：UFO [bold]{len(ufo_news)}[/bold] 条，"  # type: ignore
        f"政治危机 [bold]{len(crisis_news)}[/bold] 条（policy={policy_name}）[/green]"
    )
    console.print(f"[green]✓ 数据已保存至 {SCRAPED_FILE}[/green]")  # type: ignore
    console.print(f"[green]✓ 机制诊断已保存至 {OFFICIAL_LEAD_DIAG_FILE}[/green]")  # type: ignore
    return output


def filter_by_auth_label(items, min_confidence="MEDIUM"):
    threshold = AUTH_RANK.get(min_confidence.upper(), AUTH_RANK["MEDIUM"])  # type: ignore
    return [
        i for i in items
        if AUTH_RANK.get(i.get("authenticity", {}).get("label", "LOW"), 1) >= threshold  # type: ignore
    ]


def find_temporal_correlations(scraped_data, window_days=WINDOW_DAYS_DEFAULT, min_confidence="MEDIUM"):
    """
    在抓取数据中寻找时间关联。
    窗口为双向：UFO新闻在危机前后 window_days 天内均计入。
    """
    ufo_news = filter_by_auth_label(scraped_data.get("ufo_news", []), min_confidence=min_confidence)  # type: ignore
    crisis_news = filter_by_auth_label(scraped_data.get("crisis_news", []), min_confidence=min_confidence)  # type: ignore

    candidates = []
    for crisis in crisis_news:
        crisis_date = parse_iso_date(crisis.get("date"))  # type: ignore
        if crisis_date is None:
            continue

        for ufo in ufo_news:
            ufo_date = parse_iso_date(ufo.get("date"))  # type: ignore
            if ufo_date is None:
                continue

            gap = (ufo_date - crisis_date).days
            if abs(gap) > window_days:
                continue

            candidates.append({
                "crisis_id": crisis.get("claim_fingerprint", crisis.get("title", "")[:80]),  # type: ignore
                "ufo_id": ufo.get("claim_fingerprint", ufo.get("title", "")[:80]),  # type: ignore
                "crisis_title": crisis.get("title", ""),  # type: ignore
                "crisis_date": crisis.get("date", ""),  # type: ignore
                "crisis_score": crisis.get("authenticity", {}).get("final_score", 0),  # type: ignore
                "ufo_title": ufo.get("title", ""),  # type: ignore
                "ufo_date": ufo.get("date", ""),  # type: ignore
                "ufo_score": ufo.get("authenticity", {}).get("final_score", 0),  # type: ignore
                "gap_days": gap,
                "pattern": "事后UFO" if gap >= 0 else "事前UFO",
            })

    # Greedy 选择：优先“危机后事件 + 时间更近 + 真实性更高”，并限制重复配对。
    candidates.sort(
        key=lambda x: (
            0 if x["gap_days"] >= 0 else 1,  # type: ignore
            abs(x["gap_days"]),  # type: ignore
            -(x["ufo_score"] + x["crisis_score"]),  # type: ignore
        )
    )

    crisis_quota = defaultdict(int)
    ufo_quota = defaultdict(int)
    correlations = []
    seen_pairs = set()
    max_matches_per_crisis = 3
    max_matches_per_ufo = 2

    for c in candidates:
        pair_key = (c["crisis_id"], c["ufo_id"])  # type: ignore
        if pair_key in seen_pairs:
            continue
        if crisis_quota[c["crisis_id"]] >= max_matches_per_crisis:  # type: ignore
            continue
        if ufo_quota[c["ufo_id"]] >= max_matches_per_ufo:  # type: ignore
            continue

        seen_pairs.add(pair_key)
        crisis_quota[c["crisis_id"]] += 1  # type: ignore
        ufo_quota[c["ufo_id"]] += 1  # type: ignore
        c.pop("crisis_id", None)  # type: ignore
        c.pop("ufo_id", None)  # type: ignore
        correlations.append(c)

    correlations.sort(key=lambda x: x["gap_days"])  # type: ignore
    return correlations


def cross_check_with_history(scraped_data, window_days=WINDOW_DAYS_DEFAULT, min_confidence="MEDIUM"):
    """将新爬取数据与 events_v2.json 历史数据集中最近的危机比较"""
    if not os.path.exists(EVENTS_V2_FILE):  # type: ignore
        return []

    with open(EVENTS_V2_FILE, "r", encoding="utf-8") as f:
        historical = json.load(f)  # type: ignore

    recent_crises = []
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=180)  # type: ignore
    for case in historical.get("correlations", []):  # type: ignore
        crisis = case.get("crisis", {})  # type: ignore
        d = parse_iso_date(crisis.get("date"))  # type: ignore
        if d and d >= cutoff:
            recent_crises.append(crisis)

    if not recent_crises:
        return []

    ufo_news = filter_by_auth_label(scraped_data.get("ufo_news", []), min_confidence=min_confidence)  # type: ignore
    candidates = []

    for crisis in recent_crises:
        crisis_date = parse_iso_date(crisis.get("date"))  # type: ignore
        if crisis_date is None:
            continue

        for ufo in ufo_news:
            ufo_date = parse_iso_date(ufo.get("date"))  # type: ignore
            if ufo_date is None:
                continue

            gap = (ufo_date - crisis_date).days
            if abs(gap) > window_days:
                continue

            candidates.append({
                "crisis_id": crisis.get("title", "")[:80],  # type: ignore
                "ufo_id": ufo.get("claim_fingerprint", ufo.get("title", "")[:80]),  # type: ignore
                "source": "历史数据集",
                "crisis_title": crisis.get("title", ""),  # type: ignore
                "crisis_date": crisis.get("date", ""),  # type: ignore
                "ufo_title": ufo.get("title", ""),  # type: ignore
                "ufo_date": ufo.get("date", ""),  # type: ignore
                "ufo_score": ufo.get("authenticity", {}).get("final_score", 0),  # type: ignore
                "gap_days": gap,
                "pattern": "事后UFO" if gap >= 0 else "事前UFO",
            })

    candidates.sort(
        key=lambda x: (
            0 if x["gap_days"] >= 0 else 1,  # type: ignore
            abs(x["gap_days"]),  # type: ignore
            -x.get("ufo_score", 0),  # type: ignore
        )
    )

    crisis_quota = defaultdict(int)
    ufo_quota = defaultdict(int)
    matches = []
    seen_pairs = set()
    max_matches_per_crisis = 3
    max_matches_per_ufo = 2

    for c in candidates:
        pair_key = (c["crisis_id"], c["ufo_id"])  # type: ignore
        if pair_key in seen_pairs:
            continue
        if crisis_quota[c["crisis_id"]] >= max_matches_per_crisis:  # type: ignore
            continue
        if ufo_quota[c["ufo_id"]] >= max_matches_per_ufo:  # type: ignore
            continue
        seen_pairs.add(pair_key)
        crisis_quota[c["crisis_id"]] += 1  # type: ignore
        ufo_quota[c["ufo_id"]] += 1  # type: ignore
        c.pop("crisis_id", None)  # type: ignore
        c.pop("ufo_id", None)  # type: ignore
        matches.append(c)

    matches.sort(key=lambda x: x["gap_days"])  # type: ignore
    return matches


def print_correlations(correlations, title="近期发现的关联事件"):
    if not correlations:
        console.print("[yellow]未发现近期关联事件[/yellow]")  # type: ignore
        return

    table = Table(title=title, box=box.ROUNDED, show_lines=True)  # type: ignore
    table.add_column("政治危机日期", style="red", width=12)
    table.add_column("政治危机", style="red", width=30)
    table.add_column("间隔", style="yellow", width=8, justify="center")
    table.add_column("UFO日期", style="green", width=12)
    table.add_column("UFO事件", style="green", width=30)
    table.add_column("UFO分", style="cyan", width=6, justify="right")

    for c in correlations[:20]:  # type: ignore
        gap = c["gap_days"]  # type: ignore
        gap_str = f"+{gap}天" if gap >= 0 else f"{gap}天"
        gap_color = "green" if 0 <= gap <= 30 else ("yellow" if 0 <= gap <= 60 else "dim")
        table.add_row(
            c["crisis_date"],  # type: ignore
            c["crisis_title"][:28] + ("…" if len(c["crisis_title"]) > 28 else ""),  # type: ignore
            f"[{gap_color}]{gap_str}[/{gap_color}]",
            c["ufo_date"],  # type: ignore
            c["ufo_title"][:28] + ("…" if len(c["ufo_title"]) > 28 else ""),  # type: ignore
            str(c.get("ufo_score", 0)),  # type: ignore
        )

    console.print(table)  # type: ignore


def parse_args():
    parser = argparse.ArgumentParser(description="抓取新闻并做严格真实性评估")
    parser.add_argument(
        "--policy",
        choices=POLICY_CHOICES,
        default=None,
        help="评审策略：strict | strict-balanced | lenient（默认 strict）",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="兼容参数：等价于 --policy lenient",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=LOOKBACK_DAYS_DEFAULT,
        help=f"新闻回看窗口天数（默认 {LOOKBACK_DAYS_DEFAULT}）",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help=f"历史全窗口模式（等价于 --lookback-days {FULL_HISTORY_LOOKBACK_DAYS}）",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=WINDOW_DAYS_DEFAULT,
        help=f"危机-UFO关联窗口天数（默认 {WINDOW_DAYS_DEFAULT}）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS_DEFAULT,
        help=f"并发抓取线程数（默认 {MAX_WORKERS_DEFAULT}）",
    )
    parser.add_argument(
        "--max-items-per-source",
        type=int,
        default=MAX_ITEMS_PER_SOURCE,
        help=f"每个来源最多读取条目数（默认 {MAX_ITEMS_PER_SOURCE}）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.policy:
        policy_name = args.policy
    elif args.lenient:
        policy_name = POLICY_LENIENT
    else:
        policy_name = POLICY_STRICT

    lookback_days = FULL_HISTORY_LOOKBACK_DAYS if args.full_history else args.lookback_days
    lookback_days = max(7, lookback_days)
    max_items_per_source = max(5, args.max_items_per_source)

    console.print(  # type: ignore
        f"[bold cyan]开始抓取最新新闻数据（policy={policy_name}, "
        f"lookback={lookback_days}天, max_items_per_source={max_items_per_source}）...[/bold cyan]\n"
    )

    data = scrape_all(
        policy_name=policy_name,
        lookback_days=lookback_days,
        max_workers=max(1, args.max_workers),
        max_items_per_source=max_items_per_source,
    )

    ufo_count = len(data.get("ufo_news", []))  # type: ignore
    crisis_count = len(data.get("crisis_news", []))  # type: ignore
    console.print(  # type: ignore
        f"\n[bold]本次通过真实性审查：UFO [green]{ufo_count}[/green] 条 / "  # type: ignore
        f"政治危机 [red]{crisis_count}[/red] 条[/bold]"  # type: ignore
    )

    live_corr = find_temporal_correlations(data, window_days=max(1, args.window_days), min_confidence="MEDIUM")
    if live_corr:
        print_correlations(live_corr, title="当前抓取数据内部关联（MEDIUM+）")

    history_matches = cross_check_with_history(
        data,
        window_days=max(1, args.window_days),
        min_confidence="MEDIUM",
    )
    if history_matches:
        post = [m for m in history_matches if m["gap_days"] >= 0]  # type: ignore
        console.print(  # type: ignore
            f"\n[bold yellow]历史危机 × 最新UFO新闻 交叉验证："  # type: ignore
            f"发现 [white]{len(history_matches)}[/white] 个关联对"
            f"（危机后 UFO 出现：[green]{len(post)}[/green] 个）[/bold yellow]"
        )
        print_correlations(post, title="历史危机→当前UFO新闻波（MEDIUM+）")
    else:
        console.print("\n[yellow]无历史危机与当前新闻形成关联（近180天）[/yellow]")  # type: ignore


if __name__ == "__main__":
    main()
