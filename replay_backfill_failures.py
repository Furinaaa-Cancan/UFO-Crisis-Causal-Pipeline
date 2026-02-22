"""
历史回填失败分段重放器

用途：
1) 从 data/historical_backfill_runs.json 读取失败分段
2) 自动按 query + date range 逐段重试 historical_backfill.py
3) 产出重放审计报告，便于后续持续补齐
"""
# pyre-ignore-all-errors
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

from utils import parse_date  # type: ignore[import]


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_HISTORY_FILE = DATA_DIR / "historical_backfill_runs.json"
DEFAULT_REPORT_FILE = DATA_DIR / "historical_backfill_replay_report.json"
DEFAULT_REPORT_HISTORY_FILE = DATA_DIR / "historical_backfill_replay_runs.json"
BACKFILL_REPORT_FILE = DATA_DIR / "historical_backfill_report.json"
BACKFILL_SCRIPT = BASE_DIR / "historical_backfill.py"
PYTHON_BIN = BASE_DIR / ".venv" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="重放 historical_backfill 的失败分段")
    p.add_argument("--history-file", default=str(DEFAULT_HISTORY_FILE), help="历史回填 runs 文件")
    p.add_argument("--report-file", default=str(DEFAULT_REPORT_FILE), help="本次重放报告输出路径")
    p.add_argument("--report-history-file", default=str(DEFAULT_REPORT_HISTORY_FILE), help="重放历史累计输出路径")
    p.add_argument("--run-id", default="", help="指定回放某个 run_id；为空则用最近失败 runs")
    p.add_argument("--last-n-runs", type=int, default=1, help="未指定 run-id 时，回放最近 N 个失败 runs")
    p.add_argument("--queries", default="", help="只回放这些查询（逗号分隔）；为空表示按 run 中失败查询")
    p.add_argument("--max-chunks", type=int, default=0, help="最多回放多少个失败分段（0=不限）")
    p.add_argument("--failure-cooldown-hours", type=float, default=6.0, help="失败/部分成功分段的冷却时长（小时）")
    p.add_argument("--slice-days", type=int, default=0, help="将失败分段切成固定天数小窗口重放（0=不切）")
    p.add_argument(
        "--schedule-order",
        choices=["shortest", "longest", "none"],
        default="shortest",
        help="重放任务调度顺序：shortest 先跑短窗口，longest 先跑长窗口，none 保持字典序",
    )
    p.add_argument("--policy", default="", help="强制覆盖 policy；为空则沿用 run 的 policy")
    p.add_argument("--python-bin", default=str(PYTHON_BIN), help="用于调用 historical_backfill.py 的 Python 可执行文件")
    p.add_argument("--allow-partial", action="store_true", help="允许分段失败时继续")
    p.add_argument("--overwrite-backfill", action="store_true", help="覆盖已有 backfill 行")
    p.add_argument("--chunk-days", type=int, default=365)
    p.add_argument("--request-timeout", type=int, default=20)
    p.add_argument("--request-retries", type=int, default=2)
    p.add_argument("--rate-limit-cooldown", type=float, default=60.0)
    p.add_argument("--retry-backoff-max", type=float, default=180.0)
    p.add_argument("--google-fallback", action="store_true", help="重放时开启 Google News RSS 小窗口补抓")
    p.add_argument("--google-max-span-days", type=int, default=14, help="Google RSS 补抓最大窗口天数")
    p.add_argument("--pause-between-chunks", type=float, default=1.0)
    p.add_argument("--min-split-days", type=int, default=90)
    p.add_argument("--sleep-between-jobs", type=float, default=1.0, help="每个重放任务之间暂停秒数")
    p.add_argument("--use-env-proxy", action="store_true")
    p.add_argument("--verbose-chunks", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="只打印计划，不真正执行")
    return p.parse_args()


def load_runs(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"history file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    runs = payload.get("runs", []) if isinstance(payload, dict) else []
    if not isinstance(runs, list):
        raise ValueError("invalid history format: 'runs' must be list")
    return runs


def load_replay_runs(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []
    runs = payload.get("runs", []) if isinstance(payload, dict) else []
    return runs if isinstance(runs, list) else []


def _parse_iso_ts(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        ts = datetime.fromisoformat(raw)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def parse_query_subset(raw: str) -> set[str]:
    out = {x.strip() for x in str(raw).split(",") if x.strip()}
    return out


def select_runs_for_replay(runs: List[dict], run_id: str, last_n_runs: int) -> List[dict]:
    if run_id:
        for r in runs:
            if str(r.get("run_id", "")) == run_id:
                return [r]
        raise ValueError(f"run_id not found: {run_id}")

    failed_runs = [r for r in runs if int((r.get("stats", {}) or {}).get("failed_chunks_total", 0)) > 0]
    if not failed_runs:
        return []
    n = max(1, int(last_n_runs))
    return failed_runs[-n:]


def collect_failed_jobs(
    selected_runs: List[dict],
    selected_queries: set[str],
    max_chunks: int,
    slice_days: int,
    schedule_order: str,
) -> List[dict]:
    def split_range(start_iso: str, end_iso: str) -> List[tuple[str, str]]:
        s = parse_date(start_iso)
        e = parse_date(end_iso)
        if s > e:
            return []
        size = max(0, int(slice_days))
        if size <= 0:
            return [(s.isoformat(), e.isoformat())]
        out: List[tuple[str, str]] = []
        cur = s
        while cur <= e:
            chunk_end = min(e, cur + timedelta(days=max(1, size) - 1))  # type: ignore
            out.append((cur.isoformat(), chunk_end.isoformat()))  # type: ignore
            cur = chunk_end + timedelta(days=1)  # type: ignore
        return out

    jobs: List[dict] = []
    seen = set()
    for run in selected_runs:
        run_id = str(run.get("run_id", ""))
        policy = str(run.get("policy", "strict-balanced"))
        failed_chunks = run.get("failed_chunks", {}) or {}
        if not isinstance(failed_chunks, dict):
            continue

        for query, chunks in failed_chunks.items():
            if selected_queries and query not in selected_queries:
                continue
            if not isinstance(chunks, list):
                continue
            for c in chunks:
                start = str(c.get("start", ""))
                end = str(c.get("end", ""))
                if not start or not end:
                    continue
                try:
                    ranges = split_range(start, end)
                except Exception:
                    continue
                for s_iso, e_iso in ranges:
                    key = (query, s_iso, e_iso)
                    if key in seen:
                        continue
                    seen.add(key)
                    span_days = (parse_date(e_iso) - parse_date(s_iso)).days + 1
                    jobs.append(
                        {
                            "query": query,
                            "start": s_iso,
                            "end": e_iso,
                            "span_days": int(span_days),
                            "source_run_id": run_id,
                            "policy": policy,
                        }
                    )

    if schedule_order == "shortest":
        jobs.sort(key=lambda x: (int(x.get("span_days", 0)), x["query"], x["start"], x["end"]))  # type: ignore
    elif schedule_order == "longest":
        jobs.sort(key=lambda x: (-int(x.get("span_days", 0)), x["query"], x["start"], x["end"]))  # type: ignore
    else:
        jobs.sort(key=lambda x: (x["query"], x["start"], x["end"]))  # type: ignore
    if max_chunks > 0:
        jobs = jobs[: max_chunks]
    return jobs


def filter_jobs_by_replay_history(
    jobs: List[dict],
    replay_runs: List[dict],
    failure_cooldown_hours: float,
) -> tuple[List[dict], List[dict]]:
    latest_by_key = {}
    for run in replay_runs:
        run_ts = _parse_iso_ts(str(run.get("generated_at", "")))
        if run_ts is None:
            run_ts = _parse_iso_ts(str(run.get("updated_at", "")))
        if run_ts is None:
            continue
        for job in (run.get("jobs", []) if isinstance(run.get("jobs", []), list) else []):
            key = (
                str(job.get("query", "")),
                str(job.get("start", "")),
                str(job.get("end", "")),
                str(job.get("policy", "")),
            )
            if not all(key):
                continue
            prev = latest_by_key.get(key)
            if prev is None or run_ts > prev["ts"]:
                latest_by_key[key] = {
                    "ts": run_ts,
                    "status": str(job.get("status", "")),
                    "source_run_id": str(job.get("source_run_id", "")),
                }

    now_utc = datetime.now(timezone.utc)
    cooldown = max(0.0, float(failure_cooldown_hours))
    filtered: List[dict] = []
    skipped: List[dict] = []
    for job in jobs:
        key = (
            str(job.get("query", "")),
            str(job.get("start", "")),
            str(job.get("end", "")),
            str(job.get("policy", "")),
        )
        hist = latest_by_key.get(key)
        if hist is None:
            filtered.append(job)
            continue

        status = str(hist.get("status", ""))
        ts = hist.get("ts")
        age_h = (now_utc - ts).total_seconds() / 3600.0 if isinstance(ts, datetime) else 999999.0

        if status == "full_success":
            skipped.append(
                {
                    "query": key[0],
                    "start": key[1],
                    "end": key[2],
                    "policy": key[3],
                    "reason": "already_full_success",
                    "last_status": status,
                    "last_age_hours": round(age_h, 3),
                    "source_run_id": str(hist.get("source_run_id", "")),
                }
            )
            continue

        if status in ("partial_success", "failed") and age_h < cooldown:
            skipped.append(
                {
                    "query": key[0],
                    "start": key[1],
                    "end": key[2],
                    "policy": key[3],
                    "reason": "cooldown_active",
                    "last_status": status,
                    "last_age_hours": round(age_h, 3),
                    "source_run_id": str(hist.get("source_run_id", "")),
                }
            )
            continue

        filtered.append(job)

    return filtered, skipped


def append_report_history(path: Path, run_report: dict) -> None:
    payload = {"updated_at": datetime.now(timezone.utc).isoformat(), "runs": []}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old, dict) and isinstance(old.get("runs"), list):
                payload["runs"] = old["runs"]
        except Exception:
            payload["runs"] = []
    payload["runs"].append(run_report)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_backfill_command(args: argparse.Namespace, job: dict, policy: str) -> List[str]:
    cmd = [
        str(args.python_bin),
        str(BACKFILL_SCRIPT),
        "--start-date",
        str(job["start"]),
        "--end-date",
        str(job["end"]),
        "--queries",
        str(job["query"]),
        "--policy",
        policy,
        "--chunk-days",
        str(args.chunk_days),
        "--request-timeout",
        str(args.request_timeout),
        "--request-retries",
        str(args.request_retries),
        "--rate-limit-cooldown",
        str(args.rate_limit_cooldown),
        "--retry-backoff-max",
        str(args.retry_backoff_max),
        "--google-max-span-days",
        str(args.google_max_span_days),
        "--pause-between-chunks",
        str(args.pause_between_chunks),
        "--min-split-days",
        str(args.min_split_days),
    ]
    if args.allow_partial:
        cmd.append("--allow-partial")
    if args.overwrite_backfill:
        cmd.append("--overwrite-backfill")
    if args.google_fallback:
        cmd.append("--google-fallback")
    if args.use_env_proxy:
        cmd.append("--use-env-proxy")
    if args.verbose_chunks:
        cmd.append("--verbose-chunks")
    return cmd


def derive_job_status(returncode: int, backfill_stats: dict) -> str:
    if int(returncode) != 0:
        return "failed"
    failed_total = int((backfill_stats or {}).get("failed_chunks_total", 0))
    if failed_total == 0:
        return "full_success"
    return "partial_success"


def read_latest_backfill_report() -> dict:
    if not BACKFILL_REPORT_FILE.exists():
        return {}
    try:
        with BACKFILL_REPORT_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main() -> None:
    args = parse_args()
    history_path = Path(args.history_file)
    report_path = Path(args.report_file)
    report_history_path = Path(args.report_history_file)

    runs = load_runs(history_path)
    replay_runs = load_replay_runs(report_history_path)
    selected_runs = select_runs_for_replay(runs, args.run_id.strip(), args.last_n_runs)
    selected_queries = parse_query_subset(args.queries)
    raw_jobs = collect_failed_jobs(
        selected_runs=selected_runs,
        selected_queries=selected_queries,
        max_chunks=args.max_chunks,
        slice_days=args.slice_days,
        schedule_order=args.schedule_order,
    )
    jobs, skipped_history = filter_jobs_by_replay_history(
        jobs=raw_jobs,
        replay_runs=replay_runs,
        failure_cooldown_hours=args.failure_cooldown_hours,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    replay_report = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "history_file": str(history_path),
        "selected_run_ids": [str(r.get("run_id", "")) for r in selected_runs],
        "selected_queries": sorted(selected_queries),
        "failure_cooldown_hours": float(args.failure_cooldown_hours),
        "slice_days": int(args.slice_days),
        "schedule_order": str(args.schedule_order),
        "jobs_total_before_history_filter": len(raw_jobs),
        "jobs_total": len(jobs),
        "skipped_by_history": len(skipped_history),
        "skipped_jobs": skipped_history,
        "jobs": [],
        "stats": {
            "executed": 0,
            "dry_run": bool(args.dry_run),
            "full_success": 0,
            "partial_success": 0,
            "failed": 0,
        },
    }

    if not jobs:
        replay_report["note"] = "no_failed_chunks_found"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(replay_report, f, ensure_ascii=False, indent=2)
        append_report_history(report_history_path, replay_report)
        print("=== Replay Backfill Failures ===")
        print("jobs_total: 0")
        print(f"jobs_total_before_history_filter: {replay_report['jobs_total_before_history_filter']}")
        print(f"skipped_by_history: {replay_report['skipped_by_history']}")
        print("note: no_failed_chunks_found_or_all_skipped_by_history")
        print(f"[输出] {report_path}")
        print(f"[输出] {report_history_path}")
        return

    for i, job in enumerate(jobs, start=1):
        policy = args.policy.strip() or str(job.get("policy", "strict-balanced"))
        cmd = build_backfill_command(args, job, policy)
        print(
            f"[replay] job {i}/{len(jobs)} query={job['query']} range={job['start']}~{job['end']} "
            f"policy={policy}",
            flush=True,
        )
        if args.dry_run:
            replay_report["jobs"].append(  # type: ignore
                {
                    "job_index": i,
                    "query": job["query"],
                    "start": job["start"],
                    "end": job["end"],
                    "policy": policy,
                    "source_run_id": job["source_run_id"],
                    "status": "dry_run",
                    "command": cmd,
                }
            )
            continue

        proc = subprocess.run(cmd, cwd=str(BASE_DIR))
        replay_report["stats"]["executed"] += 1  # type: ignore

        latest = read_latest_backfill_report()
        stats = latest.get("stats", {}) if isinstance(latest, dict) else {}
        status = derive_job_status(proc.returncode, stats if isinstance(stats, dict) else {})
        if status == "failed":
            replay_report["stats"]["failed"] += 1  # type: ignore
        elif status == "full_success":
            replay_report["stats"]["full_success"] += 1  # type: ignore
        else:
            replay_report["stats"]["partial_success"] += 1  # type: ignore

        replay_report["jobs"].append(  # type: ignore
            {
                "job_index": i,
                "query": job["query"],
                "start": job["start"],
                "end": job["end"],
                "policy": policy,
                "source_run_id": job["source_run_id"],
                "status": status,
                "returncode": proc.returncode,
                "backfill_stats": stats,
            }
        )
        time.sleep(max(0.0, float(args.sleep_between_jobs)))

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(replay_report, f, ensure_ascii=False, indent=2)
    append_report_history(report_history_path, replay_report)

    print("=== Replay Backfill Failures ===")
    print(f"jobs_total_before_history_filter: {replay_report['jobs_total_before_history_filter']}")
    print(f"jobs_total: {replay_report['jobs_total']}")
    print(f"skipped_by_history: {replay_report['skipped_by_history']}")
    print(f"executed: {replay_report['stats']['executed']}")
    print(f"full_success: {replay_report['stats']['full_success']}")
    print(f"partial_success: {replay_report['stats']['partial_success']}")
    print(f"failed: {replay_report['stats']['failed']}")
    print(f"[输出] {report_path}")
    print(f"[输出] {report_history_path}")


if __name__ == "__main__":
    main()
