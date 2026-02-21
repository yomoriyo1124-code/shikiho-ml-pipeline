# 四季報オンライン 株価上昇率予測パイプライン 完全版

> **File:** docs/plans/03_add_nextday_targets.md
> **Date:** 2026-02-21
> **Status:** Draft（実装待ち）
> **前身計画:** docs/plans/01_shikiho_ml_pipeline.md → docs/plans/02_pipeline_v2.md

---

## 1. 背景 / ゴール

**背景:**
四季報オンラインが週次で更新する業績修正情報は、発表当日の後場に株価を動かすことが経験則として知られている。
この動きを **機械学習で予測** し、毎週「上昇期待銘柄ランキング」を自動生成するパイプラインを構築する。

**ゴール:**
週次CSVを `run_pipeline.py` に渡すだけで、以下が自動実行される：

```
データ拡充（Agent1）→ 複数モデル比較・予測（Agent2）→ レポート生成（Agent3）→ 翌週フィードバック生成（Agent4）
```

**株価への織り込みパターン仮説:**
- **当日吸収型** — 後場中に完全に折り込まれ、終値に反映（12:30→終値が大きい）
- **翌朝浸透型** — 引け後に個人・海外投資家が情報を消化し、翌日ギャップとして現れる（翌日始値が大きい）

4ターゲットを比較することで「どのタイミングに上昇が集中するか」の特性も分析できる。

---

## 2. スコープ

### In（やること）
- [ ] Agent 1: 週次CSVに株価（1h足）・基本情報・翌日始値を付与
- [ ] Agent 2: 4ターゲット × 複数MLモデルを比較し最良モデルで予測。精度不足なら5クラス分類
- [ ] Agent 3: 予測精度・ランキング・吸収パターン分析のHTMLレポート生成
- [ ] Agent 4: レポートを解析し Agent1・Agent2 への翌週フィードバックJSONを生成
- [ ] 4エージェントをスクリプトで直列パイプライン化
- [ ] 1h足ベースで過去データ遡及取得（最大約730日）

### Out（やらないこと）
- リアルタイム自動売買・投資推奨
- 四季報オンラインへの自動スクレイピング（CSVは手動ダウンロード継続）
- 有料API利用
- 深層学習（LSTM等）— 初期フェーズでは対象外（データ蓄積後に検討）

---

## 3. 現状把握

### CSVカラム（確認済）

| カラム名 | 内容 | 備考 |
|----------|------|------|
| 証券コード | 銘柄コード | yfinance: `{コード}.T` |
| 社名 | 会社名 | |
| 市場 | 東P/東G/東S | カテゴリ特徴量 |
| 経常修正率(%) | 経常利益前回比修正率 | 主要特徴量 |
| 純利益修正率(%) | 純利益前回比修正率 | 主要特徴量 |
| 時価総額(億円) | 時価総額 | log変換検討 |
| 更新日 | 四季報更新日 | 発表とのタイムラグ計算 |
| 備考 | 備考欄 | NaN多い |
| 発表 | 株価取得対象日 | |

### yfinance 1h足 検証結果（7203.T, 発表日: 2026-02-19）

| 日時 | Open | Close | 意味 |
|------|------|-------|------|
| 2026-02-19 09:00 | — | — | 発表日始値 |
| 2026-02-19 11:00 | — | **3766** | **前場引け（11:30）に対応** |
| 2026-02-19 12:00 | **3763** | 3755 | **後場始値（12:30）に対応** |
| 2026-02-19 15:00 | — | 3764 | 発表日終値 |
| **2026-02-20 09:00** | **3671** | — | **翌営業日始値 ✓** |

**価格取得方法（確定）:**

| 価格 | 取得方法 | 備考 |
|------|---------|------|
| `price_open` | 発表日 09:00バーの `Open` | 発表日始値 |
| `price_1130` | 発表日 11:00バーの `Close` | 前場引け近似 |
| `price_1230` | 発表日 12:00バーの `Open` | 後場始値近似 |
| `price_close` | 日次データの `Close` | 精度優先 |
| `price_nextopen` | 翌営業日 09:00バー（最初の行）の `Open` | 週末・祝日自動対応 |

**制約:**
- 1h足は最長730日（2023年2月〜）。それ以前は取得不可
- 英字コードを含む場合（例: 135A）も `.T` 付きで yfinance は対応

### 4ターゲット変数（最終版）

| # | ターゲット変数 | 計算式 | 投資イメージ |
|---|--------------|--------|------------|
| 1 | `target_1130_close` | `(price_close - price_1130) / price_1130 × 100` | 昼休み中に仕込んで当日引け |
| 2 | `target_1230_close` | `(price_close - price_1230) / price_1230 × 100` | 後場寄りで仕込んで当日引け |
| 3 | `target_1130_nextopen` | `(price_nextopen - price_1130) / price_1130 × 100` | 昼休み中に仕込んで翌朝成り行き売り |
| 4 | `target_1230_nextopen` | `(price_nextopen - price_1230) / price_1230 × 100` | 後場寄りで仕込んで翌朝成り行き売り |

---

## 4. プロジェクト構造

```
四季報/
├── date/
│   ├── row_date/                  # 元CSV（手動配置）
│   │   └── 四季報_YYMMDD.csv
│   └── processed/                 # 拡充済みCSV（蓄積）
│       ├── 四季報_YYMMDD_processed.csv
│       ├── 四季報_YYMMDD_predicted.csv
│       └── master.csv             # 全週マージ
├── models/
│   └── model_{target}_{YYMMDD}.pkl   # 学習済みモデル（4ターゲット×最良モデル）
├── reports/
│   └── report_YYMMDD.html         # 週次レポート
├── feedback/
│   └── feedback_YYMMDD.json       # Agent4出力（翌週用）
├── agents/
│   ├── agent1_data_updater.py
│   ├── agent2_ml_predictor.py
│   ├── agent3_reporter.py
│   └── agent4_feedback.py
├── run_pipeline.py
└── requirements.txt
```

**モデルファイル命名例:**
```
models/model_target_1130_close_260102.pkl
models/model_target_1230_close_260102.pkl
models/model_target_1130_nextopen_260102.pkl
models/model_target_1230_nextopen_260102.pkl
```

---

## 5. 作業ステップ

### Phase A: 環境構築

**ステップ 1: requirements.txt 作成**

```
yfinance>=0.2.40
pandas>=2.0
scikit-learn>=1.4
lightgbm>=4.0
xgboost>=2.0
catboost>=1.2
optuna>=3.0
matplotlib>=3.8
seaborn>=0.13
jinja2>=3.1
joblib>=1.3
```

**ステップ 2: ディレクトリ構造の作成**

```bash
mkdir -p date/row_date date/processed models reports feedback agents
```

**ステップ 3: 1h足取得プロトタイプ検証**

`四季報_260102.csv` の上位5銘柄（英字コード `135A` を含む）で以下の5価格が正しく取得できることを確認：
- `price_open` / `price_1130` / `price_1230` / `price_close` / `price_nextopen`

---

### Phase B: Agent 1 — データ更新エージェント

**ファイル:** `agents/agent1_data_updater.py`

**入力:**
- `date/row_date/四季報_YYMMDD.csv`（週次CSV）
- `feedback/feedback_前週YYMMDD.json`（Agent4からの指示、初回は空）

**出力:**
- `date/processed/四季報_YYMMDD_processed.csv`
- `date/processed/master.csv`（全週蓄積、重複チェック付き）

**ステップ 4: 株価取得モジュール（5価格）**

```python
import yfinance as yf
from datetime import timedelta

def fetch_prices(code: str, announce_date) -> dict:
    ticker = yf.Ticker(f"{code}.T")

    # 発表日の1h足
    df_1h = ticker.history(
        start=announce_date,
        end=announce_date + timedelta(days=1),
        interval="1h"
    )
    price_open     = df_1h.iloc[0]['Open']
    price_1130     = df_1h.loc[df_1h.index.hour == 11, 'Close'].iloc[-1]
    price_1230     = df_1h.loc[df_1h.index.hour == 12, 'Open'].iloc[0]

    # 発表日終値（日次データで取得、精度優先）
    df_daily       = ticker.history(start=announce_date, end=announce_date + timedelta(days=1))
    price_close    = df_daily['Close'].iloc[0]

    # 翌営業日始値（最大7日先まで取得で週末・祝日に自動対応）
    df_next        = ticker.history(
        start=announce_date + timedelta(days=1),
        end=announce_date + timedelta(days=7),
        interval="1h"
    )
    price_nextopen = df_next.iloc[0]['Open']
    next_date      = df_next.index[0].date()

    return {
        'price_open': price_open,
        'price_1130': price_1130,
        'price_1230': price_1230,
        'price_close': price_close,
        'price_nextopen': price_nextopen,
        'next_trade_date': next_date,
    }
```

**ステップ 5: 財務指標取得モジュール（Ticker.info）**

| 項目 | yfinanceキー | 備考 |
|------|-------------|------|
| 業種 | `sector` / `industry` | 英語のままカテゴリ化 |
| PER | `trailingPE` または `forwardPE` | 異常値（負・1000超）は除外 |
| ROE | `returnOnEquity` | |
| 配当利回り | `dividendYield` | |
| β値 | `beta` | ボラティリティの代理変数 |

**ステップ 6: 派生特徴量計算**

| 特徴量 | 計算式 |
|--------|--------|
| 修正方向_経常 | `sign(経常修正率)` → -1/0/+1 |
| 修正幅（絶対値） | `abs(経常修正率)` |
| 時価総額_log | `log1p(時価総額)` |
| 更新→発表ラグ(日) | `発表日 - 更新日` |
| 発表曜日 | `発表日.dayofweek` |
| 週内累積銘柄数 | 同発表日の累積銘柄数（市場過熱度の代理） |

**ステップ 7: 4ターゲット変数計算**

```python
row['target_1130_close']    = (price_close    - price_1130) / price_1130    * 100
row['target_1230_close']    = (price_close    - price_1230) / price_1230    * 100
row['target_1130_nextopen'] = (price_nextopen - price_1130) / price_1130    * 100
row['target_1230_nextopen'] = (price_nextopen - price_1230) / price_1230    * 100
```

**ステップ 8: Agent4フィードバック反映**

`feedback_前週.json` の `agent1_feedback` を読み込み、`add_features` / `remove_features` を動的に適用。

**ステップ 9: 週次ファイル保存 + master.csv への追記**

- `発表日` でユニーク確認、既存行は上書きしない
- `time.sleep(0.5)` を銘柄間に挿入（レート制限対策）
- 取得失敗銘柄はスキップして警告ログに記録

---

### Phase C: Agent 2 — ML予測エージェント

**ファイル:** `agents/agent2_ml_predictor.py`

**入力:**
- `date/processed/master.csv`
- `feedback/feedback_前週YYMMDD.json`

**出力:**
- `models/model_{target}_{YYMMDD}.pkl`（4ターゲット × 最良モデル = 4ファイル）
- `date/processed/四季報_YYMMDD_predicted.csv`
- `models/model_summary_YYMMDD.json`

**ステップ 10: モデル比較探索（4ターゲット × 5モデル）**

各ターゲットに対して以下を独立して実行：

**ベースライン（Step 10-1）:**

| モデル | ライブラリ | 特徴 |
|--------|-----------|------|
| Ridge回帰 | sklearn | 線形ベースライン |
| Random Forest | sklearn | ノンパラ・外れ値に強い |

**Gradient Boosting 比較（Step 10-2）:**

| モデル | ライブラリ | 特徴 |
|--------|-----------|------|
| **LightGBM** | lightgbm | 高速・カテゴリ変数対応・主力候補 |
| XGBoost | xgboost | LGBMと比較用 |
| CatBoost | catboost | カテゴリ特徴量に強い（業種等） |

**評価指標（回帰）:** MAE / RMSE / 方向一致率

**ステップ 11: 精度判定 → 分類切替ロジック（ターゲットごとに独立）**

```
if 方向一致率 < 55% かつ データ週数 >= 4週:
    → 5クラス分類モードへ切替
else:
    → 回帰モード継続
```

**5クラス分類（分類モード時）:**

| クラス | 条件 | ラベル |
|--------|------|--------|
| 4 | target > +2% | 強気上昇 |
| 3 | +0.5% ≤ target ≤ +2% | 軽度上昇 |
| 2 | -0.5% < target < +0.5% | 横ばい |
| 1 | -2% ≤ target ≤ -0.5% | 軽度下落 |
| 0 | target < -2% | 強気下落 |

分類評価: F1（macro）、混同行列。クラス不均衡対策: `class_weight='balanced'`

**ステップ 12: 時系列交差検証**

`TimeSeriesSplit(n_splits=5)` で全モデルを比較し、最良モデルを選択（ターゲットごとに独立）。

**ステップ 13: Optuna ハイパーパラメータ最適化**

- 対象: 最良モデルのみ
- 試行回数: 初期 50回、データ蓄積後（8週以上）に 100回

**ステップ 14: 予測・モデルサマリー出力**

最新週への予測を適用し、予測値＋クラス（5ランク）を出力CSV に追記。

**model_summary JSON 構造:**

```json
{
  "week": "YYMMDD",
  "targets": {
    "target_1130_close":    { "model": "lgbm", "mode": "regression", "mae": 0.8, "direction_accuracy": 0.58 },
    "target_1230_close":    { "model": "catboost", "mode": "regression", "mae": 0.7, "direction_accuracy": 0.61 },
    "target_1130_nextopen": { "model": "lgbm", "mode": "regression", "mae": 1.1, "direction_accuracy": 0.54 },
    "target_1230_nextopen": { "model": "xgboost", "mode": "regression", "mae": 1.0, "direction_accuracy": 0.56 }
  }
}
```

**ステップ 15: Agent4フィードバック反映**

`feedback_前週.json` の `agent2_feedback` を読み込み、推奨モデル・特徴量・ハイパーパラメータヒントを適用。

---

### Phase D: Agent 3 — レポート分析エージェント

**ファイル:** `agents/agent3_reporter.py`

**入力:**
- `date/processed/四季報_YYMMDD_predicted.csv`
- `models/model_summary_YYMMDD.json`

**出力:** `reports/report_YYMMDD.html`（Jinja2テンプレート使用）

**ステップ 16: レポート構成（6 + 1セクション）**

| # | セクション | 内容 |
|---|-----------|------|
| 1 | 精度サマリー | 4ターゲット × MAE/RMSE/方向一致率（または F1）の表、前週比 |
| 2 | 特徴量重要度 | Top 15 グラフ（4ターゲット別） |
| 3 | モデル比較表 | 全候補の精度一覧 |
| 4 | 上昇期待ランキング | Top 30（銘柄コード・社名・予測値・ランク）4ターゲット別タブ |
| 5 | 前週予測 vs 実績 | 乖離大きい銘柄の分析 |
| 6 | **上昇吸収パターン分析** | 当日吸収型 vs 翌朝浸透型の比較（下記詳細） |
| 7 | 次週改善サジェスト | Agent4への入力メモ |

**セクション6 詳細（上昇吸収パターン分析）:**

```
当日吸収型（price_close > price_nextopen）: xx% の銘柄
翌朝浸透型（price_nextopen > price_close）: xx% の銘柄
平均的な当日吸収割合: xx%
```

→「四季報情報はいつ株価に織り込まれるか」を可視化するセクション。

---

### Phase E: Agent 4 — フィードバックエージェント

**ファイル:** `agents/agent4_feedback.py`

**入力:**
- `reports/report_YYMMDD.html`
- `models/model_summary_YYMMDD.json`

**出力:** `feedback/feedback_YYMMDD.json`

**ステップ 17: フィードバックJSON構造（4ターゲット対応版）**

```json
{
  "week": "YYMMDD",
  "agent1_feedback": {
    "add_features": ["feature_name: 追加理由"],
    "remove_features": ["feature_name: 除外理由"],
    "data_quality_issues": ["issue description"]
  },
  "agent2_feedback": {
    "target_1130_close": {
      "recommended_model": "lgbm_regression",
      "switch_to_classification": false,
      "feature_importance_hints": {
        "top_features": ["経常修正率", "..."],
        "low_importance": ["削除候補feature"]
      },
      "hyperparameter_hints": {
        "lgbm_num_leaves": 31,
        "lgbm_learning_rate": 0.05
      }
    },
    "target_1230_close":    { "..." : "..." },
    "target_1130_nextopen": { "..." : "..." },
    "target_1230_nextopen": { "..." : "..." },
    "best_target": "target_1230_close"
  },
  "general_notes": "自由記述メモ"
}
```

**ステップ 18: 自動分析ルール（ヒューリスティック）**

- 特徴量重要度 < 0.5% → `remove_features` に追加
- 前週 MAE > 閾値 → `switch_to_classification: true`
- 乖離大銘柄に共通する属性（業種・市場等）があれば `add_features` に提案
- 4ターゲット中で最も方向一致率が高いものを `best_target` に設定

---

### Phase F: パイプライン化

**ファイル:** `run_pipeline.py`

**ステップ 19: run_pipeline.py 作成**

```bash
# 使用例（毎週の実行コマンド）
python run_pipeline.py date/row_date/四季報_260102.csv
```

実行順序:
```
Agent1（data_updater）
    ↓
Agent2（ml_predictor）
    ↓
Agent3（reporter）
    ↓
Agent4（feedback）
```

- 各エージェントは失敗時にエラーログを出力して停止（次エージェントへは進まない）
- フィードバックJSONの読み込みは Agent1・Agent2 が実行開始時に自動検索（最新の `feedback_*.json` を使用）

---

## 6. 成果物（Deliverables）

| 成果物 | ファイル / 場所 | 形式 | 完了条件 |
|--------|----------------|------|---------|
| 環境定義 | `requirements.txt` | TXT | `pip install -r requirements.txt` が通る |
| データ更新Agent | `agents/agent1_data_updater.py` | Python | 発表日の5価格（open/1130/1230/close/nextopen）が取得され処理済みCSVが生成される |
| ML予測Agent | `agents/agent2_ml_predictor.py` | Python | 4ターゲット×5モデル比較が実行され、最良モデルで予測値付きCSVが生成される |
| レポートAgent | `agents/agent3_reporter.py` | Python | 7セクション（吸収パターン分析含む）のHTMLレポートが出力される |
| フィードバックAgent | `agents/agent4_feedback.py` | Python | 4ターゲット別 `feedback_YYMMDD.json` が生成され、次週のAgent1・2に読み込まれる |
| パイプラインスクリプト | `run_pipeline.py` | Python | 1コマンドで4エージェントが完走する |
| 蓄積マスタCSV | `date/processed/master.csv` | CSV | 週次追記で重複なく蓄積される |
| 週次レポート | `reports/report_YYMMDD.html` | HTML | ブラウザで開ける完成レポート |
| フィードバックJSON | `feedback/feedback_YYMMDD.json` | JSON | 翌週実行時にAgent1・2が正常に読み込める |

---

## 7. リスク / 代替案

| リスク | 影響 | 確率 | 代替案 / 緩和策 |
|--------|------|------|----------------|
| 1h足の11:30/12:30近似がズレる（前場引け気配） | Medium | Low | 11:00バーClose / 12:00バーOpen の正確性は 7203.T で検証済み。Phase A で複数銘柄も確認 |
| 1h足が730日を超えた日付は取得不可 | Low | Low | 当面 ~2年分確保できるため許容 |
| 翌日が連続休場（年末年始・GW）で翌日始値が数日後になる | Low | Low | `timedelta(days=7)` で最大1週間先まで取得するため自動対応 |
| Agent2の学習時間が4ターゲット分かかる | Medium | Medium | 各ターゲット並列実行（`joblib.Parallel`）で対応。Optuna試行数を動的調整 |
| 5クラスのクラス不均衡（横ばいが多数） | Medium | High | `class_weight='balanced'`。評価は macro-F1 を使用 |
| Agent4の自動フィードバックが誤った指示を出す | Medium | Medium | JSONはテキストで確認可能。ユーザーが毎週レビューして手動上書きできる設計にする |
| yfinance API レート制限 / タイムアウト | Medium | Medium | `time.sleep(0.5)` を銘柄間に挿入。バッチ分割で再試行ロジック実装 |
| 特定銘柄でティッカーが見つからない | Low | Medium | 取得失敗銘柄はスキップして警告ログに記録。後で手動確認 |
| 翌日始値ターゲットの予測精度が当日終値より著しく低い | Low | Medium | 精度比較でフィードバック。Agent4が精度不足のターゲットを除外提案 |

---

## 8. 実装順序と次アクション

### 実装順序

```
Phase A（環境・プロトタイプ検証）
    ↓
Phase B（Agent1）→ 動作確認
    ↓
Phase C（Agent2）→ 動作確認
    ↓
Phase D（Agent3）→ 動作確認
    ↓
Phase E（Agent4）→ 動作確認
    ↓
Phase F（パイプライン統合）→ 全体通しテスト
```

### データ戦略

| 時期 | 状況 | 重点 |
|------|------|------|
| 初回（1週目） | `四季報_260102.csv` のみ | Agent1で発表日の5価格を取得・検証 |
| 2〜3週目 | 2〜3週分蓄積 | Agent2のモデル比較を本格化（データ少ないため Ridge/RF中心） |
| 4週目以降 | 4週分以上 | 精度判定・分類切替が機能し始める |
| 6〜8週目以降 | 十分な蓄積 | Agent4のフィードバックループが有効化 |

### 次の1手（Phase A 開始）

> `requirements.txt` 作成 + `pip install` で環境構築完了。
> 続けて `四季報_260102.csv` の上位5銘柄（英字コード `135A` を含む）で、
> **発表日の5価格（open/1130/1230/close/nextopen）** がすべて正しく取得できることをプロトタイプスクリプトで確認する。
