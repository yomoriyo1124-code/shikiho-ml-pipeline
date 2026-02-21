#!/usr/bin/env python3
"""Agent 4: フィードバックエージェント

入力:
  - reports/report_YYMMDD.html          (参照のみ。主要データは JSON から取得)
  - models/model_summary_YYMMDD.json

出力:
  - feedback/feedback_YYMMDD.json

使用方法:
  python agents/agent4_feedback.py                           # 最新ファイルを自動検出
  python agents/agent4_feedback.py <report_html> <summary_json>
"""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── パス定義 ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "date" / "processed"
FEEDBACK_DIR = BASE_DIR / "feedback"

ENCODING = "utf-8-sig"

# ── ターゲット定義 ─────────────────────────────────────────────────────────────
TARGETS = [
    "target_1130_close",
    "target_1230_close",
    "target_1130_nextopen",
    "target_1230_nextopen",
]

TARGET_LABELS = {
    "target_1130_close": "11:30→終値",
    "target_1230_close": "12:30→終値",
    "target_1130_nextopen": "11:30→翌寄",
    "target_1230_nextopen": "12:30→翌寄",
}

# ── 閾値 ───────────────────────────────────────────────────────────────────────
LOW_IMPORTANCE_THRESHOLD = 0.5    # 重要度(%)この値未満は低重要度
DA_CLASSIFICATION_THRESHOLD = 0.55  # DA < この値 → 分類モード推奨
DA_REGRESSION_THRESHOLD = 0.55     # DA >= この値 → 回帰モード推奨
MAE_THRESHOLDS = {                  # 回帰モード時の MAE 警告閾値
    "target_1130_close": 1.5,
    "target_1230_close": 1.5,
    "target_1130_nextopen": 2.0,
    "target_1230_nextopen": 2.0,
}
NAN_RATE_WARN = 0.30                # NaN率がこの値を超えたら警告


# ══════════════════════════════════════════════════════════════════════════════
# ヒューリスティック分析
# ══════════════════════════════════════════════════════════════════════════════

def _best_target(summary: dict) -> str:
    """4ターゲット中で方向一致率が最も高いものを返す"""
    return max(
        TARGETS,
        key=lambda t: summary["targets"].get(t, {}).get("direction_accuracy") or 0.0,
    )


def _analyze_target(tgt: str, info: dict) -> dict:
    """1ターゲット分の Agent2 フィードバックを生成"""
    mode = info["mode"]
    da = info.get("direction_accuracy") or 0.0
    top_features: dict = info.get("top_features", {})
    all_models: dict = info.get("all_models_reg", {})
    best_params: dict = info.get("best_params", {})
    current_model = info.get("model", "lgbm")

    # ── 分類/回帰モード推奨 ──────────────────────────────────────────────────
    use_classification = da < DA_CLASSIFICATION_THRESHOLD

    # ── 推奨モデル（all_models_reg の DA 最優先、分類時は ridge を除外）────
    # ridge は線形モデルのため 5クラス分類には不向き → 分類推奨時は除外
    CLASSIFICATION_EXCLUDED = {"ridge"}
    if all_models:
        candidates = {
            m: v for m, v in all_models.items()
            if not (use_classification and m in CLASSIFICATION_EXCLUDED)
        } or all_models  # 全除外になった場合は元に戻す
        rec_model = max(
            candidates,
            key=lambda m: candidates[m].get("direction_accuracy", 0.0),
        )
    else:
        rec_model = current_model

    # 推奨モード文字列
    next_mode = "classification" if use_classification else "regression"
    recommended_model = f"{rec_model}_{next_mode}"

    # ── 特徴量重要度ヒント ───────────────────────────────────────────────────
    top5 = list(top_features.keys())[:5]
    low_importance = [k for k, v in top_features.items() if v < LOW_IMPORTANCE_THRESHOLD]

    # ── ハイパーパラメータヒント（モデル名をプレフィックスとして付与）────────
    hp_hints = {f"{current_model}_{k}": v for k, v in best_params.items()}

    result: dict = {
        "recommended_model": recommended_model,
        "switch_to_classification": use_classification,
        "feature_importance_hints": {
            "top_features": top5,
            "low_importance": low_importance,
        },
        "hyperparameter_hints": hp_hints,
    }

    # ── MAE 警告（回帰モードのみ）──────────────────────────────────────────
    mae = info.get("mae")
    if mae is not None and mode == "regression":
        threshold = MAE_THRESHOLDS.get(tgt, 1.5)
        if mae > threshold:
            result["mae_warning"] = (
                f"MAE={mae:.4f} が閾値 {threshold} を超過 → 特徴量・モデル見直し推奨"
            )

    return result


def _analyze_agent1(summary: dict, predicted_csv_path: Path | None) -> dict:
    """Agent1 向けフィードバック生成"""
    # ── 3ターゲット以上で低重要度な特徴量を除外候補に ───────────────────────
    low_counter: Counter = Counter()
    for tgt in TARGETS:
        feats = summary["targets"].get(tgt, {}).get("top_features", {})
        for fname, fval in feats.items():
            if fval < LOW_IMPORTANCE_THRESHOLD:
                low_counter[fname] += 1

    remove_features = [
        f"{feat}: {cnt}ターゲットで重要度 {LOW_IMPORTANCE_THRESHOLD}% 未満"
        for feat, cnt in low_counter.most_common()
        if cnt >= 3
    ]

    # ── predicted CSV を読んでデータ品質チェック + 乖離大銘柄分析 ──────────
    data_quality_issues: list[str] = []
    add_features: list[str] = []

    if predicted_csv_path and predicted_csv_path.exists():
        try:
            df = pd.read_csv(predicted_csv_path, encoding=ENCODING)

            # NaN 率が高い特徴量列を特定
            non_feature_cols = set(TARGETS) | {
                c for c in df.columns if c.startswith("pred_") or c.startswith("rank_")
            }
            nan_rates = df.isnull().mean()
            for col in df.columns:
                if col not in non_feature_cols and nan_rates[col] > NAN_RATE_WARN:
                    data_quality_issues.append(
                        f"{col}: NaN率 {nan_rates[col]:.1%} → データ取得精度の確認推奨"
                    )

            # 乖離大銘柄における市場・業種の傾向分析（実績データがある場合）
            best_tgt = summary.get("best_target", TARGETS[0])
            rank_col = f"rank_class_{best_tgt}"
            if best_tgt in df.columns and rank_col in df.columns:
                valid = df[df[best_tgt].notna()].copy()
                if len(valid) >= 10:
                    bins = [-999, -2.0, -0.5, 0.5, 2.0, 999]
                    labels = [0, 1, 2, 3, 4]
                    valid["_actual_class"] = pd.cut(
                        valid[best_tgt], bins=bins, labels=labels
                    ).astype(float)
                    valid["_class_diff"] = (
                        valid["_actual_class"] - valid[rank_col]
                    ).abs()

                    large_gap = valid[valid["_class_diff"] >= 2]
                    if len(large_gap) >= 5 and "市場" in large_gap.columns:
                        market_dist = large_gap["市場"].value_counts()
                        top_market = market_dist.index[0]
                        # 数値エンコード済みの場合は文字列として有意でないのでスキップ
                        if isinstance(top_market, str) and len(top_market) >= 2:
                            pct = market_dist.iloc[0] / len(large_gap)
                            if pct >= 0.5:
                                add_features.append(
                                    f"市場セグメント別交互作用特徴量: "
                                    f"乖離大銘柄({len(large_gap)}件)の {pct:.0%} が {top_market} 銘柄"
                                )

        except Exception as e:
            data_quality_issues.append(f"predicted CSV 読み込みエラー: {e}")

    return {
        "add_features": add_features,
        "remove_features": remove_features,
        "data_quality_issues": data_quality_issues,
    }


def _general_notes(summary: dict) -> str:
    """全体サマリーと次週に向けたコメントを生成"""
    week = summary.get("week", "?")
    best_tgt = summary.get("best_target", "?")
    best_label = TARGET_LABELS.get(best_tgt, best_tgt)

    targets_data = summary.get("targets", {})
    modes = [targets_data[t]["mode"] for t in TARGETS if t in targets_data]
    cls_count = modes.count("classification")
    reg_count = modes.count("regression")

    das = [
        targets_data[t].get("direction_accuracy") or 0.0
        for t in TARGETS if t in targets_data
    ]
    avg_da = sum(das) / len(das) if das else 0.0

    lines = [
        f"週 {week} の自動分析完了。",
        f"ベストターゲット: {best_label}（{best_tgt}）。",
        f"モード内訳: 分類 {cls_count} ターゲット / 回帰 {reg_count} ターゲット。",
        f"平均方向一致率: {avg_da:.1%}。",
    ]

    if avg_da < 0.50:
        lines.append(
            "全体的に精度が低い水準。データ週数の蓄積が最優先改善策。"
            "特徴量エンジニアリングより先に週数を増やすことを推奨。"
        )
    elif avg_da < DA_CLASSIFICATION_THRESHOLD:
        lines.append(
            "精度は改善傾向だが分類閾値未満。"
            "週数蓄積と特徴量エンジニアリングを並行して進めること。"
        )
    else:
        lines.append(
            "方向一致率が閾値を超過。"
            "回帰モードへの移行を積極的に検討する段階。"
        )

    return " ".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════════════════════

def main(report_html: str | None = None, summary_json: str | None = None) -> str:
    # ── 入力ファイル解決 ───────────────────────────────────────────────────────
    if summary_json is None:
        jsons = sorted(MODELS_DIR.glob("model_summary_*.json"), reverse=True)
        if not jsons:
            print("[Agent4][ERROR] model_summary JSON が見つかりません")
            sys.exit(1)
        summary_json_path = jsons[0]
    else:
        summary_json_path = Path(summary_json)

    if report_html is None:
        htmls = sorted(REPORTS_DIR.glob("report_*.html"), reverse=True)
        report_html_path = htmls[0] if htmls else None
    else:
        report_html_path = Path(report_html)

    print(f"[Agent4] モデルサマリ: {summary_json_path}")
    if report_html_path:
        print(f"[Agent4] レポートHTML: {report_html_path}")
    else:
        print("[Agent4] レポートHTML なし（model_summary のみで分析）")

    # ── モデルサマリー読み込み ────────────────────────────────────────────────
    with open(summary_json_path, encoding="utf-8") as f:
        summary: dict = json.load(f)

    week: str = summary["week"]
    print(f"[Agent4] 週: {week}")

    # ── predicted CSV を探す ──────────────────────────────────────────────────
    predicted_csvs = sorted(
        PROCESSED_DIR.glob(f"四季報_{week}_predicted.csv"), reverse=True
    )
    predicted_csv_path = predicted_csvs[0] if predicted_csvs else None
    if predicted_csv_path:
        print(f"[Agent4] 予測CSV: {predicted_csv_path}")
    else:
        print("[Agent4] 予測CSV なし（データ品質チェックをスキップ）")

    # ── Agent2 フィードバック（ターゲット別）────────────────────────────────
    best_tgt = _best_target(summary)
    agent2_feedback: dict = {}
    for tgt in TARGETS:
        if tgt in summary["targets"]:
            agent2_feedback[tgt] = _analyze_target(tgt, summary["targets"][tgt])
            label = TARGET_LABELS.get(tgt, tgt)
            info = summary["targets"][tgt]
            da = info.get("direction_accuracy") or 0.0
            mode = info["mode"]
            rec = agent2_feedback[tgt]["recommended_model"]
            print(f"[Agent4]   {label}: DA={da:.3f} ({mode}) → 推奨={rec}")

    agent2_feedback["best_target"] = best_tgt

    # ── Agent1 フィードバック ─────────────────────────────────────────────────
    agent1_feedback = _analyze_agent1(summary, predicted_csv_path)

    # ── フィードバック JSON 組み立て ──────────────────────────────────────────
    feedback = {
        "week": week,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent1_feedback": agent1_feedback,
        "agent2_feedback": agent2_feedback,
        "general_notes": _general_notes(summary),
    }

    # ── 保存 ──────────────────────────────────────────────────────────────────
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEEDBACK_DIR / f"feedback_{week}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)

    print(f"[Agent4] フィードバック出力完了: {out_path}")
    print(
        f"[Agent4] ベストターゲット: "
        f"{TARGET_LABELS.get(best_tgt, best_tgt)} ({best_tgt})"
    )
    return str(out_path)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(
        report_html=args[0] if len(args) > 0 else None,
        summary_json=args[1] if len(args) > 1 else None,
    )
