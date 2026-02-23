"""
统一研究总管道

顺序：
1) 跑双档累计（strict + strict-balanced）
2) 构建对照组面板（topics + countries）
3) 跑三类模型输出
4) 跑统一严格评审
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run_cmd(cmd: list[str]) -> int:
    print(f"[run] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=BASE_DIR)
    return proc.returncode


def validate_shock_catalog_lock(
    catalog_path: Path,
    lock_path: Path,
    shock_catalog_key: str,
) -> tuple[bool, str]:
    if not catalog_path.exists():  # type: ignore
        return False, f"catalog_missing:{catalog_path}"
    if not lock_path.exists():  # type: ignore
        return False, f"lock_missing:{lock_path}"

    try:
        with catalog_path.open("r", encoding="utf-8") as f:  # type: ignore
            catalog = json.load(f)  # type: ignore
    except Exception as e:
        return False, f"catalog_invalid_json:{e}"
    try:
        with lock_path.open("r", encoding="utf-8") as f:  # type: ignore
            lock = json.load(f)  # type: ignore
    except Exception as e:
        return False, f"lock_invalid_json:{e}"

    if not isinstance(catalog, dict):
        return False, "catalog_not_object"
    if not isinstance(lock, dict):
        return False, "lock_not_object"

    catalog_sig = str(catalog.get("catalog_signature_sha256", "") or "")
    lock_sig = str(lock.get("catalog_signature_sha256", "") or "")
    if not catalog_sig:
        return False, "catalog_signature_missing"
    if not lock_sig:
        return False, "lock_signature_missing"
    if catalog_sig != lock_sig:
        return False, f"signature_mismatch:catalog={catalog_sig},lock={lock_sig}"

    key = str(shock_catalog_key or "shock_dates").strip() or "shock_dates"
    if key not in catalog:
        # 不允许静默回退，避免用错冲击键还继续跑。
        return False, f"shock_catalog_key_missing:{key}"
    return True, "ok"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="统一研究总管道")
    p.add_argument("--skip-scrape", action="store_true", help="跳过抓取")
    p.add_argument("--skip-causal", action="store_true", help="跳过因果分析")
    p.add_argument("--skip-controls", action="store_true", help="跳过对照面板构建")
    p.add_argument("--skip-models", action="store_true", help="跳过模型输出")
    p.add_argument("--only-policy", choices=["strict", "strict-balanced"], help="只跑一个策略")
    p.add_argument(
        "--model-policy",
        choices=["strict", "strict-balanced"],
        help="模型与严格评审使用的 policy（默认跟随 --only-policy，否则 strict-balanced）",
    )
    p.add_argument("--min-days", type=int, default=180)
    p.add_argument("--min-shocks", type=int, default=12)
    p.add_argument("--min-observed-ratio", type=float, default=0.85)
    p.add_argument("--max-missing-streak", type=int, default=30)
    p.add_argument(
        "--controls-lookback-days",
        type=int,
        default=-1,
        help="<=0 自动按 causal_panel 全时段构建对照面板；>0 使用回看天数",
    )
    p.add_argument(
        "--shock-catalog-file",
        default="",
        help="可选冲击日目录（传给 panel_pipeline 与模型）；为空则仅使用 events_v2 主轨",
    )
    p.add_argument(
        "--shock-catalog-key",
        default="shock_dates",
        help="冲击目录字段名（默认 shock_dates；可切到 shock_dates_nonoverlap_30d 等）",
    )
    p.add_argument(
        "--shock-lock-file",
        default="data/crisis_shock_catalog_lock.json",
        help="冲击目录锁文件路径（默认 data/crisis_shock_catalog_lock.json）",
    )
    p.add_argument(
        "--skip-shock-lock-check",
        action="store_true",
        help="跳过冲击目录锁一致性检查（仅探索场景）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable

    policies = [args.only_policy] if args.only_policy else ["strict", "strict-balanced"]
    model_policy = args.model_policy or args.only_policy or "strict-balanced"

    shock_catalog_raw = str(args.shock_catalog_file or "").strip()  # type: ignore
    if shock_catalog_raw and not args.skip_shock_lock_check:  # type: ignore
        lock_raw = str(args.shock_lock_file or "data/crisis_shock_catalog_lock.json").strip()  # type: ignore
        lock_path = Path(lock_raw)
        if not lock_path.is_absolute():  # type: ignore
            lock_path = BASE_DIR / lock_path
        ok, reason = validate_shock_catalog_lock(
            catalog_path=Path(shock_catalog_raw),  # type: ignore
            lock_path=lock_path,
            shock_catalog_key=str(args.shock_catalog_key or "shock_dates"),
        )
        if not ok:
            print(f"[error] shock catalog lock check failed: {reason}")
            raise SystemExit(3)
        print(f"[info] shock catalog lock check passed: {lock_path}")

    if args.skip_causal and not args.skip_scrape:
        print("[warn] --skip-causal 已开启：本次抓取结果不会写入 causal_panel（仅更新 scraped_news）。")

    for policy in policies:
        cmd = [
            py,
            "panel_pipeline.py",
            "--policy",
            policy,
            "--min-days",
            str(args.min_days),
            "--min-shocks",
            str(args.min_shocks),
            "--min-observed-ratio",
            str(args.min_observed_ratio),
        ]
        if args.skip_scrape:
            cmd.append("--skip-scrape")
        if args.skip_causal:
            cmd.append("--skip-causal")
        if str(args.shock_catalog_file or "").strip():  # type: ignore
            cmd.extend(["--shock-catalog-file", str(args.shock_catalog_file)])  # type: ignore
            cmd.extend(["--shock-catalog-key", str(args.shock_catalog_key or "shock_dates")])  # type: ignore
        code = run_cmd(cmd)
        if code != 0:
            raise SystemExit(code)

    if not args.skip_controls:
        controls_cmd = [py, "control_panel_builder.py", "--lookback-days", str(args.controls_lookback_days)]
        if args.skip_scrape:
            # Keep --skip-scrape semantically offline by default while still writing a full zero-filled country panel.
            controls_cmd.append("--offline-zero-fill-countries")
            print("[info] --skip-scrape 已开启：control_panel_builder 使用离线国家零值网格，不触发 country RSS 抓取。")

        code = run_cmd(controls_cmd)
        if code != 0:
            raise SystemExit(code)

    if not args.skip_models:
        for script in (
            "model_did.py",
            "model_event_study.py",
            "model_synth_control.py",
            "model_causal_ml.py",
        ):
            cmd = [py, script, "--policy", model_policy]
            if str(args.shock_catalog_file or "").strip():  # type: ignore
                cmd.extend(["--shock-catalog-file", str(args.shock_catalog_file)])  # type: ignore
                cmd.extend(["--shock-catalog-key", str(args.shock_catalog_key or "shock_dates")])  # type: ignore
            code = run_cmd(cmd)
            if code != 0:
                raise SystemExit(code)

    code = run_cmd(
        [
            py,
            "strict_reviewer.py",
            "--expected-policy",
            model_policy,
            "--min-observed-ratio",
            str(args.min_observed_ratio),
            "--max-missing-streak",
            str(args.max_missing_streak),
        ]
    )
    if code != 0:
        raise SystemExit(code)

    print("\n=== Unified Pipeline Done ===")


if __name__ == "__main__":
    main()
