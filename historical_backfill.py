"""
历史面板回填（GDELT Timeline）

目标：
1) 拉取更早年份（默认 2017-01-01 至今）的日度新闻计数
2) 回填到 data/causal_panel.json，提升 observed_days 与 shock_days
3) 明确标记来源为 gdelt_timeline_backfill，避免与实时抓取语义混淆
"""
# pyre-ignore-all-errors
from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import requests  # type: ignore

from utils import parse_date  # type: ignore[import]


BASE_DIR = Path(__file__).resolve().parent  # type: ignore
DATA_DIR = BASE_DIR / "data"
PANEL_FILE = DATA_DIR / "causal_panel.json"
REPORT_FILE = DATA_DIR / "historical_backfill_report.json"
REPORT_HISTORY_FILE = DATA_DIR / "historical_backfill_runs.json"
GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
REQUEST_TIMEOUT = 120
REQUEST_RETRIES = 3
CHUNK_DAYS = 5000
PAUSE_BETWEEN_CHUNKS = 5.2
MIN_SPLIT_DAYS = 90
RATE_LIMIT_COOLDOWN = 45.0
RETRY_BACKOFF_MAX = 120.0
BACKFILL_TAG = "gdelt_timeline_backfill_v1"

# 查询尽量短，避免语法复杂导致接口波动。
UFO_QUERY = '(ufo OR uap OR "unidentified aerial" OR extraterrestrial) sourcecountry:us sourcelang:english'
CRISIS_QUERY = (
    '(indictment OR impeach OR scandal OR "special counsel" OR subpoena OR '
    '"classified documents" OR corruption OR "federal charges") '
    'sourcecountry:us sourcelang:english'
)
CTRL_ECONOMY_QUERY = '(economy OR inflation OR tariff OR recession OR gdp) sourcecountry:us sourcelang:english'
CTRL_SECURITY_QUERY = '(war OR iran OR russia OR china OR missile OR defense) sourcecountry:us sourcelang:english'
CTRL_IMMIGRATION_QUERY = '(immigration OR border OR migrant OR asylum OR deportation) sourcecountry:us sourcelang:english'

QUERY_MAP = {
    "ufo": UFO_QUERY,
    "crisis": CRISIS_QUERY,
    "control_economy": CTRL_ECONOMY_QUERY,
    "control_security": CTRL_SECURITY_QUERY,
    "control_immigration": CTRL_IMMIGRATION_QUERY,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="历史面板回填（GDELT Timeline）")
    p.add_argument("--start-date", default="2017-01-01", help="回填起始日期（YYYY-MM-DD）")
    p.add_argument("--end-date", default=date.today().isoformat(), help="回填结束日期（YYYY-MM-DD）")
    p.add_argument("--policy", choices=["strict", "strict-balanced"], default="strict-balanced")
    p.add_argument("--overwrite-backfill", action="store_true", help="覆盖已有 backfill 行")
    p.add_argument("--chunk-days", type=int, default=CHUNK_DAYS, help=f"GDELT 分段天数（默认 {CHUNK_DAYS}）")
    p.add_argument("--request-timeout", type=int, default=REQUEST_TIMEOUT, help=f"单次请求超时秒数（默认 {REQUEST_TIMEOUT}）")
    p.add_argument("--request-retries", type=int, default=REQUEST_RETRIES, help=f"单段重试次数（默认 {REQUEST_RETRIES}）")
    p.add_argument(
        "--rate-limit-cooldown",
        type=float,
        default=RATE_LIMIT_COOLDOWN,
        help=f"遇到 429 时的最小冷却秒数（默认 {RATE_LIMIT_COOLDOWN}）",
    )
    p.add_argument(
        "--retry-backoff-max",
        type=float,
        default=RETRY_BACKOFF_MAX,
        help=f"重试退避最大秒数（默认 {RETRY_BACKOFF_MAX}）",
    )
    p.add_argument("--pause-between-chunks", type=float, default=PAUSE_BETWEEN_CHUNKS, help="分段请求间隔秒数")
    p.add_argument(
        "--min-split-days",
        type=int,
        default=MIN_SPLIT_DAYS,
        help=f"失败分段自动二分重试的最小天数阈值（默认 {MIN_SPLIT_DAYS}）",
    )
    p.add_argument("--allow-partial", action="store_true", help="分段失败时跳过该段继续（推荐）")
    p.add_argument("--skip-zero-days", action="store_true", help="跳过全零天（默认保留）")
    p.add_argument("--use-env-proxy", action="store_true", help="允许 requests 使用系统代理（默认关闭，降低代理握手阻塞风险）")
    p.add_argument("--verbose-chunks", action="store_true", help="输出每个分段请求与失败重试日志")
    p.add_argument(
        "--queries",
        default="ufo,crisis,control_economy,control_security,control_immigration",
        help="本次抓取的查询集合（逗号分隔）。例如：ufo,crisis",
    )
    return p.parse_args()


def parse_selected_queries(raw: str) -> List[str]:
    selected_queries = [x.strip() for x in str(raw).split(",") if x.strip()]
    invalid = [x for x in selected_queries if x not in QUERY_MAP]
    if invalid:
        raise ValueError(f"未知 queries: {invalid}；可选={sorted(QUERY_MAP.keys())}")
    if not selected_queries:
        raise ValueError("queries 不能为空")
    return selected_queries


def _to_gdelt_dt(d: date, end_of_day: bool = False) -> str:
    if end_of_day:
        return d.strftime("%Y%m%d") + "235959"
    return d.strftime("%Y%m%d") + "000000"


def split_date_range(start: date, end: date) -> tuple[tuple[date, date], tuple[date, date]]:
    span_days = (end - start).days + 1
    if span_days <= 1:
        return (start, end), (end, end)
    left_days = span_days // 2
    left_end = start + timedelta(days=max(0, left_days - 1))  # type: ignore
    right_start = left_end + timedelta(days=1)  # type: ignore
    return (start, left_end), (right_start, end)


def _compute_retry_sleep_seconds(
    attempt: int,
    is_429: bool,
    rate_limit_cooldown: float,
    retry_backoff_max: float,
) -> float:
    base = 1.2 * (2 ** max(0, attempt - 1))
    sleep_s = min(float(retry_backoff_max), float(base))
    if is_429:
        sleep_s = max(float(rate_limit_cooldown), sleep_s)
    return max(0.0, sleep_s)


def _request_timeline_payload(
    query: str,
    start: date,
    end: date,
    mode: str,
    request_timeout: int,
    request_retries: int,
    rate_limit_cooldown: float,
    retry_backoff_max: float,
    use_env_proxy: bool,
    verbose_chunks: bool,
) -> tuple[dict | None, str]:
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "startdatetime": _to_gdelt_dt(start),
        "enddatetime": _to_gdelt_dt(end, end_of_day=True),
    }
    last_err = ""
    connect_timeout = max(5, min(20, int(request_timeout)))
    read_timeout = max(5, int(request_timeout))
    for attempt in range(1, max(1, request_retries) + 1):
        try:
            with requests.Session() as session:
                session.trust_env = bool(use_env_proxy)
                resp = session.get(
                    GDELT_ENDPOINT,
                    params=params,
                    timeout=(connect_timeout, read_timeout),
                )  # type: ignore
            if resp.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests", response=resp)
            resp.raise_for_status()
            payload = resp.json()  # type: ignore
            return payload, ""
        except Exception as e:
            last_err = f"attempt={attempt}: {e}"
            is_429 = isinstance(e, requests.HTTPError) and getattr(getattr(e, "response", None), "status_code", None) == 429
            sleep_s = _compute_retry_sleep_seconds(
                attempt=attempt,
                is_429=is_429,
                rate_limit_cooldown=rate_limit_cooldown,
                retry_backoff_max=retry_backoff_max,
            )
            if verbose_chunks:
                err_label = "rate_limited_429" if is_429 else "request_error"
                print(
                    f"[backfill] retry {start.isoformat()} -> {end.isoformat()} mode={mode} "
                    f"attempt={attempt}/{max(1, request_retries)} sleep={sleep_s:.1f}s type={err_label}",
                    flush=True,
                )
            time.sleep(sleep_s)
    return None, last_err or "unknown_error"


def _parse_timeline_payload(payload: dict) -> Dict[str, int]:
    out: Dict[str, int] = {}
    timeline = payload.get("timeline", [])  # type: ignore
    data = timeline[0].get("data", []) if timeline else []  # type: ignore
    for row in data:
        s = str(row.get("date", ""))  # type: ignore
        if not s:
            continue
        day = s[:8]
        if len(day) != 8:
            continue
        iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        val = row.get("value", row.get("norm", 0))  # type: ignore
        try:
            out[iso] = int(float(val or 0))
        except Exception:
            out[iso] = 0
    return out


def _expand_failed_days(failed: List[dict], lower: date, upper: date) -> set[str]:
    out: set[str] = set()
    for item in failed:
        try:
            s = parse_date(str(item.get("start", "")))
            e = parse_date(str(item.get("end", "")))
        except Exception:
            continue
        if e < lower or s > upper:
            continue
        cur = max(s, lower)
        end = min(e, upper)
        while cur <= end:
            out.add(cur.isoformat())
            cur += timedelta(days=1)  # type: ignore
    return out


def _fetch_chunk_with_split(
    query: str,
    start: date,
    end: date,
    request_timeout: int,
    request_retries: int,
    rate_limit_cooldown: float,
    retry_backoff_max: float,
    pause_between_chunks: float,
    min_split_days: int,
    use_env_proxy: bool,
    verbose_chunks: bool,
) -> tuple[Dict[str, int], List[dict]]:
    modes = ("TimelineVolRaw", "TimelineVol")
    mode_errors = []
    for mode in modes:
        payload, err = _request_timeline_payload(
            query=query,
            start=start,
            end=end,
            mode=mode,
            request_timeout=request_timeout,
            request_retries=request_retries,
            rate_limit_cooldown=rate_limit_cooldown,
            retry_backoff_max=retry_backoff_max,
            use_env_proxy=use_env_proxy,
            verbose_chunks=verbose_chunks,
        )
        if payload is not None:
            return _parse_timeline_payload(payload), []
        mode_errors.append(f"{mode}: {err}")

    span_days = (end - start).days + 1
    if span_days > max(1, int(min_split_days)):
        if verbose_chunks:
            print(f"[backfill] split {start.isoformat()} -> {end.isoformat()} (span_days={span_days})", flush=True)
        (l_start, l_end), (r_start, r_end) = split_date_range(start, end)
        left_counts, left_failed = _fetch_chunk_with_split(
            query=query,
            start=l_start,
            end=l_end,
            request_timeout=request_timeout,
            request_retries=request_retries,
            rate_limit_cooldown=rate_limit_cooldown,
            retry_backoff_max=retry_backoff_max,
            pause_between_chunks=pause_between_chunks,
            min_split_days=min_split_days,
            use_env_proxy=use_env_proxy,
            verbose_chunks=verbose_chunks,
        )
        time.sleep(max(0.0, pause_between_chunks))
        right_counts, right_failed = _fetch_chunk_with_split(
            query=query,
            start=r_start,
            end=r_end,
            request_timeout=request_timeout,
            request_retries=request_retries,
            rate_limit_cooldown=rate_limit_cooldown,
            retry_backoff_max=retry_backoff_max,
            pause_between_chunks=pause_between_chunks,
            min_split_days=min_split_days,
            use_env_proxy=use_env_proxy,
            verbose_chunks=verbose_chunks,
        )
        merged = left_counts
        merged.update(right_counts)
        return merged, left_failed + right_failed

    if verbose_chunks:
        print(
            f"[backfill] fail_leaf {start.isoformat()} -> {end.isoformat()} err={' | '.join(mode_errors)}",
            flush=True,
        )
    return {}, [{
        "start": start.isoformat(),  # type: ignore
        "end": end.isoformat(),  # type: ignore
        "error": " | ".join(mode_errors),
    }]


def fetch_timeline_counts(
    query: str,
    start: date,
    end: date,
    chunk_days: int,
    request_timeout: int,
    request_retries: int,
    rate_limit_cooldown: float,
    retry_backoff_max: float,
    pause_between_chunks: float,
    min_split_days: int,
    use_env_proxy: bool,
    verbose_chunks: bool,
) -> tuple[Dict[str, int], List[dict]]:
    out: Dict[str, int] = {}
    failed_chunks: List[dict] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=max(1, chunk_days) - 1))  # type: ignore
        if verbose_chunks:
            print(f"[backfill] chunk {cursor.isoformat()} -> {chunk_end.isoformat()}", flush=True)
        counts, failed = _fetch_chunk_with_split(
            query=query,
            start=cursor,
            end=chunk_end,
            request_timeout=request_timeout,
            request_retries=request_retries,
            rate_limit_cooldown=rate_limit_cooldown,
            retry_backoff_max=retry_backoff_max,
            pause_between_chunks=pause_between_chunks,
            min_split_days=min_split_days,
            use_env_proxy=use_env_proxy,
            verbose_chunks=verbose_chunks,
        )
        out.update(counts)
        failed_chunks.extend(failed)  # type: ignore
        if verbose_chunks:
            print(
                f"[backfill] chunk_done {cursor.isoformat()} -> {chunk_end.isoformat()} days={len(counts)} failed={len(failed)}",
                flush=True,
            )

        time.sleep(max(0.0, pause_between_chunks))
        cursor = chunk_end + timedelta(days=1)  # type: ignore

    return out, failed_chunks


def load_panel() -> dict:
    if not PANEL_FILE.exists():  # type: ignore
        return {"meta": {"version": 1}, "rows": []}
    with PANEL_FILE.open("r", encoding="utf-8") as f:  # type: ignore
        obj = json.load(f)  # type: ignore
    obj.setdefault("meta", {"version": 1})  # type: ignore
    obj.setdefault("rows", [])  # type: ignore
    return obj


def should_replace_row(old_row: dict, overwrite_backfill: bool) -> bool:
    source = str(old_row.get("data_source", ""))  # type: ignore
    if source.startswith("gdelt_timeline_backfill"):
        return overwrite_backfill
    # 非 backfill 行默认不覆盖（保护实时抓取快照）
    return False


def build_row(
    day_iso: str,
    policy: str,
    ufo_count: int,
    crisis_count: int,
    ctrl_economy: int,
    ctrl_security: int,
    ctrl_immigration: int,
) -> dict:
    control_total = int(ctrl_economy + ctrl_security + ctrl_immigration)
    accepted_events = int(ufo_count + crisis_count)
    denom = max(1, accepted_events)
    control_density = control_total / float(denom)

    return {
        "date": day_iso,
        "policy": policy,
        "date_scope": "run_day_only",
        "updated_at": datetime.now(timezone.utc).isoformat(),  # type: ignore
        "ufo_count": int(ufo_count),
        "crisis_count": int(crisis_count),
        "rejected_count": 0,
        "accepted_events": accepted_events,
        "window_ufo_count": int(ufo_count),
        "window_crisis_count": int(crisis_count),
        "window_rejected_count": 0,
        "window_accepted_events": accepted_events,
        "raw_items": accepted_events,
        "ufo_score_mean": 0.0,
        "crisis_score_mean": 0.0,
        "control_scope": "accepted_only",
        "control_economy": int(ctrl_economy),
        "control_security": int(ctrl_security),
        "control_immigration": int(ctrl_immigration),
        "control_total": control_total,
        "control_density_accepted": round(control_density, 6),  # type: ignore
        "control_total_rejected_audit": 0,
        "data_source": BACKFILL_TAG,
    }


def append_report_history(run_report: dict) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),  # type: ignore
        "runs": [],
    }
    if REPORT_HISTORY_FILE.exists():  # type: ignore
        try:
            with REPORT_HISTORY_FILE.open("r", encoding="utf-8") as f:  # type: ignore
                old = json.load(f)  # type: ignore
            if isinstance(old, dict) and isinstance(old.get("runs"), list):
                payload["runs"] = old["runs"]  # type: ignore
        except Exception:
            payload["runs"] = []

    payload["runs"].append(run_report)  # type: ignore
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()  # type: ignore

    with REPORT_HISTORY_FILE.open("w", encoding="utf-8") as f:  # type: ignore
        json.dump(payload, f, ensure_ascii=False, indent=2)  # type: ignore


def main() -> None:
    args = parse_args()
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if start > end:
        raise SystemExit("start-date 不能晚于 end-date")

    try:
        selected_queries = parse_selected_queries(args.queries)
    except ValueError as e:
        raise SystemExit(str(e))

    print(f"[backfill] selected queries: {', '.join(selected_queries)}")
    series_map = {k: {} for k in QUERY_MAP}
    failed_chunks = {k: [] for k in QUERY_MAP}
    for key in selected_queries:
        print(f"[backfill] fetch {key} timeline: {start.isoformat()} -> {end.isoformat()}")
        counts, failed = fetch_timeline_counts(
            QUERY_MAP[key],  # type: ignore
            start,
            end,
            chunk_days=args.chunk_days,
            request_timeout=args.request_timeout,
            request_retries=args.request_retries,
            rate_limit_cooldown=args.rate_limit_cooldown,
            retry_backoff_max=args.retry_backoff_max,
            pause_between_chunks=args.pause_between_chunks,
            min_split_days=args.min_split_days,
            use_env_proxy=args.use_env_proxy,
            verbose_chunks=args.verbose_chunks,
        )
        series_map[key] = counts  # type: ignore
        failed_chunks[key] = failed  # type: ignore

    ufo = series_map["ufo"]  # type: ignore
    crisis = series_map["crisis"]  # type: ignore
    ctrl_e = series_map["control_economy"]  # type: ignore
    ctrl_s = series_map["control_security"]  # type: ignore
    ctrl_i = series_map["control_immigration"]  # type: ignore

    total_failed = sum(len(v) for v in failed_chunks.values())  # type: ignore
    if total_failed > 0 and not args.allow_partial:
        raise SystemExit(f"存在失败分段 {total_failed} 个；可加 --allow-partial 继续。")

    panel = load_panel()
    rows = panel.get("rows", [])  # type: ignore
    by_key = {}
    for idx, row in enumerate(rows):
        k = (row.get("date"), row.get("policy"))  # type: ignore
        by_key[k] = (idx, row)

    inserted = 0
    updated = 0
    skipped_existing = 0
    skipped_empty = 0
    skipped_failed = 0
    failed_day_sets = {
        key: _expand_failed_days(failed_chunks.get(key, []), start, end)  # type: ignore
        for key in selected_queries
    }

    d = start
    while d <= end:
        day_iso = d.isoformat()  # type: ignore
        # 分段失败视为缺失观测，不能按 0 写入，否则会制造伪零值。
        if any(day_iso in failed_day_sets.get(key, set()) for key in selected_queries):
            skipped_failed += 1
            d += timedelta(days=1)  # type: ignore
            continue
        u = int(ufo.get(day_iso, 0))
        c = int(crisis.get(day_iso, 0))
        ce = int(ctrl_e.get(day_iso, 0))
        cs = int(ctrl_s.get(day_iso, 0))
        ci = int(ctrl_i.get(day_iso, 0))

        # 默认保留全零日作为有效观测；仅在显式要求时跳过。
        if (u + c + ce + cs + ci) == 0 and args.skip_zero_days:
            skipped_empty += 1
            d += timedelta(days=1)  # type: ignore
            continue

        new_row = build_row(day_iso, args.policy, u, c, ce, cs, ci)
        key = (day_iso, args.policy)
        if key in by_key:
            idx, old = by_key[key]
            if should_replace_row(old, args.overwrite_backfill):
                rows[idx] = new_row  # type: ignore
                updated += 1
            else:
                skipped_existing += 1
        else:
            rows.append(new_row)  # type: ignore
            inserted += 1
        d += timedelta(days=1)  # type: ignore

    rows.sort(key=lambda x: (x.get("date", ""), x.get("policy", "")))  # type: ignore
    panel["rows"] = rows  # type: ignore
    panel["meta"]["updated_at"] = datetime.now(timezone.utc).isoformat()  # type: ignore

    with PANEL_FILE.open("w", encoding="utf-8") as f:  # type: ignore
        json.dump(panel, f, ensure_ascii=False, indent=2)  # type: ignore

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),  # type: ignore
        "policy": args.policy,
        "start_date": start.isoformat(),  # type: ignore
        "end_date": end.isoformat(),  # type: ignore
        "data_source": BACKFILL_TAG,
        "queries": {
            "ufo": UFO_QUERY,
            "crisis": CRISIS_QUERY,
            "control_economy": CTRL_ECONOMY_QUERY,
            "control_security": CTRL_SECURITY_QUERY,
            "control_immigration": CTRL_IMMIGRATION_QUERY,
        },
        "selected_queries": selected_queries,
        "stats": {
            "inserted_rows": inserted,
            "updated_rows": updated,
            "skipped_existing_rows": skipped_existing,
            "skipped_all_zero_days": skipped_empty,
            "skipped_failed_days": skipped_failed,
            "ufo_nonzero_days": sum(1 for v in ufo.values() if v > 0),  # type: ignore
            "crisis_nonzero_days": sum(1 for v in crisis.values() if v > 0),  # type: ignore
            "failed_chunks_total": total_failed,
        },
        "failed_chunks": failed_chunks,
        "notes": [
            "本回填基于 GDELT 日度计数，适用于样本量扩充与稳健性分析，不替代事件级人工核查。",
            "默认不覆盖实时抓取行（非 backfill 数据源）。",
            "失败分段会按缺失观测跳过，不会以零值写入面板。",
        ],
    }
    with REPORT_FILE.open("w", encoding="utf-8") as f:  # type: ignore
        json.dump(report, f, ensure_ascii=False, indent=2)  # type: ignore
    append_report_history(report)

    print("=== Historical Backfill Done ===")
    print(f"policy: {args.policy}")
    print(f"inserted_rows: {inserted}")
    print(f"updated_rows: {updated}")
    print(f"skipped_existing_rows: {skipped_existing}")
    print(f"skipped_all_zero_days: {skipped_empty}")
    print(f"skipped_failed_days: {skipped_failed}")
    print(f"failed_chunks_total: {total_failed}")
    print(f"[输出] {PANEL_FILE}")
    print(f"[输出] {REPORT_FILE}")
    print(f"[输出] {REPORT_HISTORY_FILE}")


if __name__ == "__main__":
    main()
