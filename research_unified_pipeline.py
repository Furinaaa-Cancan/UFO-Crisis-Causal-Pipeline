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
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run_cmd(cmd: list[str]) -> int:
    print(f"[run] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=BASE_DIR)
    return proc.returncode


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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable

    policies = [args.only_policy] if args.only_policy else ["strict", "strict-balanced"]
    model_policy = args.model_policy or args.only_policy or "strict-balanced"

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
            code = run_cmd([py, script, "--policy", model_policy])
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
