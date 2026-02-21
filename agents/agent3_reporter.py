#!/usr/bin/env python3
"""Agent 3: レポート分析エージェント

入力:
  - date/processed/四季報_YYMMDD_predicted.csv
  - models/model_summary_YYMMDD.json

出力:
  - reports/report_YYMMDD.html

使用方法:
  python agents/agent3_reporter.py                          # 最新ファイルを自動検出
  python agents/agent3_reporter.py <predicted_csv> <summary_json>
"""

import base64
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── パス定義 ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "date" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

ENCODING = "utf-8-sig"

# ── 日本語フォント設定（Windows: Yu Gothic / Meiryo、なければデフォルト）──────
def _setup_font():
    import matplotlib.font_manager as fm
    candidates = ["Yu Gothic", "Meiryo", "MS Gothic", "IPAexGothic", "Noto Sans CJK JP"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            return name
    return None

FONT_NAME = _setup_font()


# ══════════════════════════════════════════════════════════════════════════════
# ユーティリティ
# ══════════════════════════════════════════════════════════════════════════════

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _html_table(rows: list[dict], highlight_col: str = "") -> str:
    if not rows:
        return '<p class="no-data">データなし</p>'
    keys = list(rows[0].keys())
    ths = "".join(f"<th>{k}</th>" for k in keys)
    tbody = ""
    for row in rows:
        tds = ""
        for k in keys:
            val = row[k]
            cls = ' class="highlight"' if k == highlight_col else ""
            tds += f"<td{cls}>{val}</td>"
        tbody += f"<tr>{tds}</tr>"
    return f"<table><thead><tr>{ths}</tr></thead><tbody>{tbody}</tbody></table>"


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ══════════════════════════════════════════════════════════════════════════════
# セクション別生成
# ══════════════════════════════════════════════════════════════════════════════

TARGET_LABELS = {
    "target_1130_close": "11:30→終値",
    "target_1230_close": "12:30→終値",
    "target_1130_nextopen": "11:30→翌寄",
    "target_1230_nextopen": "12:30→翌寄",
}

CLASS_LABELS = {4: "強気上昇", 3: "軽度上昇", 2: "横ばい", 1: "軽度下落", 0: "強気下落"}


def sec1_accuracy(summary: dict) -> list[dict]:
    """精度サマリー表を生成"""
    best_target = summary["best_target"]
    rows = []
    for tgt, info in summary["targets"].items():
        mode = info["mode"]
        da = info["direction_accuracy"]
        row = {
            "ターゲット": TARGET_LABELS.get(tgt, tgt),
            "モード": "分類" if mode == "classification" else "回帰",
            "使用モデル": info["model"].upper(),
            "方向一致率": _pct(da) if da is not None else "-",
            "F1 (macro)": f"{info['f1_macro']:.3f}" if mode == "classification" else "-",
            "MAE": f"{info['mae']:.4f}" if info.get("mae") else "-",
            "Best": "★" if tgt == best_target else "",
        }
        rows.append(row)
    return rows


def sec2_feature_importance(summary: dict) -> dict[str, str]:
    """特徴量重要度グラフ（Base64埋め込み）"""
    imgs = {}
    for tgt, info in summary["targets"].items():
        feats = info.get("top_features", {})
        if not feats:
            continue
        names = list(feats.keys())[:15]
        vals = [feats[n] for n in names]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#e74c3c" if v == max(vals) else "#3498db" for v in vals]
        ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Importance (%)")
        label = TARGET_LABELS.get(tgt, tgt)
        ax.set_title(f"特徴量重要度 Top 15 — {label}", fontsize=11)
        ax.invert_yaxis()
        plt.tight_layout()
        imgs[tgt] = _fig_to_b64(fig)
    return imgs


def sec3_model_comparison(summary: dict) -> dict[str, list[dict]]:
    """全モデル比較表"""
    result = {}
    for tgt, info in summary["targets"].items():
        all_reg = info.get("all_models_reg", {})
        best_model = info["model"]
        rows = []
        for model, metrics in all_reg.items():
            rows.append({
                "モデル": model.upper(),
                "MAE": f"{metrics['mae']:.4f}",
                "方向一致率": _pct(metrics["direction_accuracy"]),
                "Best": "★" if model == best_model else "",
            })
        # MAE昇順でソート
        rows.sort(key=lambda r: float(r["MAE"]))
        result[tgt] = rows
    return result


def sec4_ranking(df: pd.DataFrame, summary: dict) -> tuple[list[dict], str]:
    """上昇期待ランキング Top 30"""
    best_target = summary["best_target"]
    rank_col = f"rank_class_{best_target}"
    pred_col = f"pred_{best_target}"

    # rank 3（軽度上昇）以上を対象
    top_df = (
        df[df[rank_col] >= 3]
        .sort_values([rank_col, pred_col], ascending=[False, False])
        .head(30)
        .copy()
    )

    rows = []
    for _, row in top_df.iterrows():
        pred_val = row.get(pred_col, "")
        if isinstance(pred_val, float):
            pred_val = f"{pred_val:.3f}"
        rows.append({
            "証券コード": int(row["証券コード"]),
            "社名": row["社名"],
            "市場": row["市場"],
            "ランク": CLASS_LABELS.get(int(row[rank_col]), str(int(row[rank_col]))),
            "予測値": pred_val,
        })
    return rows, best_target


def sec5_prev_vs_actual(df: pd.DataFrame, summary: dict) -> tuple[list[dict] | None, str | None]:
    """前週予測 vs 実績（実績がない場合はメッセージ）"""
    best_target = summary["best_target"]
    actual_col = best_target
    rank_col = f"rank_class_{best_target}"

    # 実績値が存在する行のみ
    valid = df[df[actual_col].notna()].copy()
    if len(valid) == 0:
        return None, "予測対象週のターゲット実績は未確定のため、比較データなし（翌週以降に更新されます）"

    # 実際のクラス
    bins = [-999, -2, -0.5, 0.5, 2, 999]
    labels = [0, 1, 2, 3, 4]
    valid["actual_class"] = pd.cut(valid[actual_col], bins=bins, labels=labels).astype(float)
    valid["class_diff"] = (valid["actual_class"] - valid[rank_col]).abs()

    worst = (
        valid.nlargest(10, "class_diff")
        [["証券コード", "社名", actual_col, rank_col, "actual_class", "class_diff"]]
        .copy()
    )
    rows = []
    for _, row in worst.iterrows():
        rows.append({
            "証券コード": int(row["証券コード"]),
            "社名": row["社名"],
            "実績(%)": f"{row[actual_col]:.2f}",
            "予測ランク": CLASS_LABELS.get(int(row[rank_col]), str(int(row[rank_col]))),
            "実績ランク": CLASS_LABELS.get(int(row["actual_class"]), str(int(row["actual_class"]))),
            "ランク乖離": int(row["class_diff"]),
        })
    return rows, None


def sec6_suggestions(summary: dict) -> list[str]:
    """次週改善サジェスト"""
    suggestions = []
    for tgt, info in summary["targets"].items():
        label = TARGET_LABELS.get(tgt, tgt)
        da = info.get("direction_accuracy") or 0
        mode = info["mode"]

        # 方向一致率によるモード推奨
        if mode == "classification" and da >= 0.55:
            suggestions.append(
                f"[{label}] 方向一致率が {_pct(da)} に改善 → 回帰モードへの切替を検討"
            )
        elif mode == "regression" and da < 0.52:
            suggestions.append(
                f"[{label}] 方向一致率が {_pct(da)} と低い → 分類モードへの切替を検討"
            )

        # 低重要度特徴量
        low_feats = [k for k, v in info.get("top_features", {}).items() if v < 0.5]
        if low_feats:
            suggestions.append(
                f"[{label}] 低重要度特徴量（除外候補）: {', '.join(low_feats[:5])}"
            )

    if not suggestions:
        suggestions.append("現時点で特段の改善サジェストなし。データ蓄積を継続してください。")
    return suggestions


# ══════════════════════════════════════════════════════════════════════════════
# HTML テンプレート
# ══════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>四季報予測レポート {week}</title>
<style>
  body {{
    font-family: 'Yu Gothic', 'Meiryo', 'MS Gothic', sans-serif;
    margin: 24px 32px;
    background: #f4f6f9;
    color: #2c3e50;
  }}
  h1 {{ border-bottom: 3px solid #2980b9; padding-bottom: 8px; }}
  h2 {{ color: #2980b9; margin-top: 44px; border-left: 4px solid #2980b9; padding-left: 10px; }}
  h3 {{ color: #555; margin-top: 20px; }}
  .meta {{ color: #888; font-size: 0.9em; margin-bottom: 28px; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0 20px;
    background: white;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    font-size: 0.92em;
  }}
  th {{
    background: #2980b9;
    color: white;
    padding: 8px 14px;
    text-align: left;
    white-space: nowrap;
  }}
  td {{
    padding: 6px 14px;
    border-bottom: 1px solid #eef0f2;
    white-space: nowrap;
  }}
  tr:hover {{ background: #eaf4fb; }}
  td.highlight {{ font-weight: bold; color: #c0392b; }}
  .star {{ color: #f39c12; font-weight: bold; }}
  .no-data {{ color: #aaa; font-style: italic; padding: 10px 0; }}
  ul.suggest {{ line-height: 1.9; padding-left: 20px; }}
  img {{ max-width: 720px; margin: 8px 0; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.12); }}
  .section-note {{ color: #7f8c8d; font-size: 0.88em; margin: -6px 0 8px; }}
</style>
</head>
<body>
<h1>四季報予測レポート — 週: {week}</h1>
<p class="meta">生成日時: {generated_at} ｜ ベストターゲット: <strong>{best_target_label}</strong> ｜ 銘柄数: {n_stocks}</p>

<h2>セクション 1: 精度サマリー</h2>
{sec1_html}

<h2>セクション 2: 特徴量重要度 Top 15</h2>
{sec2_html}

<h2>セクション 3: モデル比較表</h2>
{sec3_html}

<h2>セクション 4: 上昇期待ランキング Top 30（{best_target_label}）</h2>
<p class="section-note">ランク「軽度上昇」以上の銘柄を予測スコア降順で表示</p>
{sec4_html}

<h2>セクション 5: 前週予測 vs 実績</h2>
<p class="section-note">ランク乖離が大きい銘柄（Top 10）</p>
{sec5_html}

<h2>セクション 6: 次週改善サジェスト</h2>
{sec6_html}
</body>
</html>
"""


def _render_sec2(imgs: dict[str, str]) -> str:
    if not imgs:
        return '<p class="no-data">特徴量重要度データなし</p>'
    parts = []
    for tgt, b64 in imgs.items():
        label = TARGET_LABELS.get(tgt, tgt)
        parts.append(f'<h3>{label}</h3><img src="data:image/png;base64,{b64}" alt="feature importance {tgt}">')
    return "\n".join(parts)


def _render_sec3(model_cmp: dict[str, list[dict]]) -> str:
    parts = []
    for tgt, rows in model_cmp.items():
        label = TARGET_LABELS.get(tgt, tgt)
        parts.append(f"<h3>{label}</h3>" + _html_table(rows))
    return "\n".join(parts)


def _render_sec6(suggestions: list[str]) -> str:
    items = "".join(f"<li>{s}</li>" for s in suggestions)
    return f'<ul class="suggest">{items}</ul>'


# ══════════════════════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════════════════════

def main(predicted_csv: str | None = None, summary_json: str | None = None) -> str:
    # ── 入力ファイル解決 ───────────────────────────────────────────────────────
    if predicted_csv is None:
        csvs = sorted(PROCESSED_DIR.glob("四季報_*_predicted.csv"), reverse=True)
        if not csvs:
            print("[Agent3][ERROR] 予測済みCSVが見つかりません")
            sys.exit(1)
        predicted_csv_path = csvs[0]
    else:
        predicted_csv_path = Path(predicted_csv)

    if summary_json is None:
        jsons = sorted(MODELS_DIR.glob("model_summary_*.json"), reverse=True)
        if not jsons:
            print("[Agent3][ERROR] model_summary JSONが見つかりません")
            sys.exit(1)
        summary_json_path = jsons[0]
    else:
        summary_json_path = Path(summary_json)

    print(f"[Agent3] 入力CSV: {predicted_csv_path}")
    print(f"[Agent3] モデルサマリ: {summary_json_path}")

    # ── データ読み込み ─────────────────────────────────────────────────────────
    df = pd.read_csv(predicted_csv_path, encoding=ENCODING)
    with open(summary_json_path, encoding="utf-8") as f:
        summary = json.load(f)

    week = summary["week"]
    best_target = summary["best_target"]
    best_target_label = TARGET_LABELS.get(best_target, best_target)
    print(f"[Agent3] 週: {week}, 銘柄数: {len(df)}, ベストターゲット: {best_target_label}")

    # ── 各セクション生成 ───────────────────────────────────────────────────────
    s1 = sec1_accuracy(summary)
    s2 = sec2_feature_importance(summary)
    s3 = sec3_model_comparison(summary)
    s4_rows, _ = sec4_ranking(df, summary)
    s5_rows, s5_msg = sec5_prev_vs_actual(df, summary)
    s6 = sec6_suggestions(summary)

    # ── HTML 生成 ──────────────────────────────────────────────────────────────
    sec5_html = (
        f'<p class="no-data">{s5_msg}</p>' if s5_msg else _html_table(s5_rows)
    )

    html = HTML_TEMPLATE.format(
        week=week,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        best_target_label=best_target_label,
        n_stocks=len(df),
        sec1_html=_html_table(s1),
        sec2_html=_render_sec2(s2),
        sec3_html=_render_sec3(s3),
        sec4_html=_html_table(s4_rows),
        sec5_html=sec5_html,
        sec6_html=_render_sec6(s6),
    )

    # ── 保存 ──────────────────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"report_{week}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[Agent3] レポート出力完了: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(
        predicted_csv=args[0] if len(args) > 0 else None,
        summary_json=args[1] if len(args) > 1 else None,
    )
