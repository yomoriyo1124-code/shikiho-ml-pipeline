#!/usr/bin/env python3
"""四季報 ML パイプライン — 全4エージェントを直列実行

使用例:
  # 通常実行（全エージェント）
  python run_pipeline.py date/row_date/四季報_260102.csv

  # 途中から再開（Agent1 完了済みの場合）
  python run_pipeline.py --start-from agent2

  # Agent3 以降のみ再実行
  python run_pipeline.py --start-from agent3

実行順序:
  Agent1（data_updater）→ Agent2（ml_predictor）→ Agent3（reporter）→ Agent4（feedback）

各エージェントが失敗（exit code != 0）した時点でパイプラインを中断します。
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
AGENTS_DIR = BASE_DIR / "agents"

AGENT_ORDER = ["agent1", "agent2", "agent3", "agent4"]

AGENT_LABELS = {
    "agent1": "Agent1: データ更新（株価・財務・EDINET取得）",
    "agent2": "Agent2: ML予測（4ターゲット × 複数モデル比較）",
    "agent3": "Agent3: レポート生成（HTML出力）",
    "agent4": "Agent4: フィードバック生成（翌週用JSON）",
}

AGENT_SCRIPTS = {
    "agent1": AGENTS_DIR / "agent1_data_updater.py",
    "agent2": AGENTS_DIR / "agent2_ml_predictor.py",
    "agent3": AGENTS_DIR / "agent3_reporter.py",
    "agent4": AGENTS_DIR / "agent4_feedback.py",
}


# ══════════════════════════════════════════════════════════════════════════════
# ステップ実行
# ══════════════════════════════════════════════════════════════════════════════

def run_step(agent: str, extra_args: list[str] | None = None) -> float:
    """1エージェントをサブプロセスで実行。失敗時はパイプラインを即停止。経過秒を返す。"""
    label = AGENT_LABELS[agent]
    script = AGENT_SCRIPTS[agent]
    cmd = [sys.executable, str(script)] + (extra_args or [])

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"[Pipeline] {label}")
    print(f"[Pipeline] 開始: {datetime.now().strftime('%H:%M:%S')}")
    print(sep)

    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[Pipeline][ERROR] {label} が失敗しました (exit code={result.returncode})")
        print("[Pipeline] パイプラインを中断します。")
        sys.exit(result.returncode)

    print(f"\n[Pipeline] {label} 完了 ({elapsed:.1f}秒)")
    return elapsed


# ══════════════════════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="四季報 ML パイプライン（4エージェント直列実行）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help="週次CSVファイルのパス（例: date/row_date/四季報_260102.csv）。"
             " --start-from agent2 以降の場合は省略可。",
    )
    parser.add_argument(
        "--start-from",
        choices=AGENT_ORDER,
        default="agent1",
        metavar="AGENT",
        help=f"途中から再開するエージェント。選択肢: {AGENT_ORDER}  (default: agent1)",
    )
    args = parser.parse_args()

    # ── 実行するエージェントを決定 ────────────────────────────────────────────
    start_idx = AGENT_ORDER.index(args.start_from)
    agents_to_run = AGENT_ORDER[start_idx:]

    # Agent1 を実行する場合は csv_path が必須
    if "agent1" in agents_to_run:
        if args.csv_path is None:
            parser.error("agent1 を実行する場合は csv_path が必要です。")
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            print(f"[Pipeline][ERROR] CSVファイルが見つかりません: {csv_path}")
            sys.exit(1)
    else:
        csv_path = None

    # ── ヘッダー出力 ──────────────────────────────────────────────────────────
    border = "#" * 62
    print(f"\n{border}")
    print("# 四季報 ML パイプライン")
    print(f"# 開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if csv_path:
        print(f"# 入力CSV: {csv_path}")
    print(f"# 実行エージェント: {' → '.join(agents_to_run)}")
    print(border)

    # ── 各エージェントを逐次実行 ──────────────────────────────────────────────
    pipeline_start = time.time()
    timings: dict[str, float] = {}

    for agent in agents_to_run:
        extra: list[str] = []
        if agent == "agent1" and csv_path is not None:
            extra = [str(csv_path)]
        timings[agent] = run_step(agent, extra_args=extra if extra else None)

    # ── フッター出力 ──────────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    print(f"\n{border}")
    print("# パイプライン完了")
    print(f"# 終了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# 総実行時間: {total / 60:.1f}分 ({total:.0f}秒)")
    print(border)

    print("\n[エージェント別実行時間]")
    for agent, secs in timings.items():
        label = AGENT_LABELS[agent]
        print(f"  {label}: {secs / 60:.1f}分 ({secs:.0f}秒)")

    # ── 出力ファイルの場所を表示 ──────────────────────────────────────────────
    print("\n[出力ファイル]")
    _print_latest("  予測CSV  ", BASE_DIR / "date" / "processed", "四季報_*_predicted.csv")
    _print_latest("  モデルサマリ", BASE_DIR / "models", "model_summary_*.json")
    _print_latest("  レポート  ", BASE_DIR / "reports", "report_*.html")
    _print_latest("  フィードバック", BASE_DIR / "feedback", "feedback_*.json")


def _print_latest(label: str, directory: Path, pattern: str) -> None:
    """最新ファイルのパスを表示するユーティリティ"""
    files = sorted(directory.glob(pattern), reverse=True)
    if files:
        print(f"{label}: {files[0]}")
    else:
        print(f"{label}: (なし)")


if __name__ == "__main__":
    main()
