# 四季報 ML パイプライン 取扱説明書

**四季報オンラインの週次更新データから、当日・翌日の株価変動を機械学習で予測するパイプラインです。**

毎週1コマンドを実行するだけで、データ取得 → 予測 → レポート生成 → フィードバックまでが自動で完了します。

---

## 目次

1. [システム全体像](#1-システム全体像)
2. [セットアップ](#2-セットアップ)
3. [毎週の使い方](#3-毎週の使い方)
4. [各エージェント詳細](#4-各エージェント詳細)
5. [出力ファイルの見方](#5-出力ファイルの見方)
6. [ファイル構成](#6-ファイル構成)
7. [よくあるトラブルと対処法](#7-よくあるトラブルと対処法)
8. [設計上の注意事項](#8-設計上の注意事項)

---

## 1. システム全体像

```
週次CSV（手動DL）
      |
      v
 [Agent 1] データ更新
  株価5価格・財務指標・EDINET XBRL を付与
      |
      v
 [Agent 2] ML予測
  4ターゲット × 5モデル比較 → 最良モデルで予測
      |
      v
 [Agent 3] レポート生成
  精度・特徴量重要度・上昇期待ランキングを HTML 出力
      |
      v
 [Agent 4] フィードバック生成
  次週の Agent1・2 への改善ヒント JSON を出力
```

### 4つの予測ターゲット

| ターゲット | 計算式 | 投資イメージ |
|-----------|--------|------------|
| `target_1130_close` | (終値 - 11:30) / 11:30 × 100 | 昼休み仕込み → 当日引け |
| `target_1230_close` | (終値 - 12:30) / 12:30 × 100 | 後場寄り仕込み → 当日引け |
| `target_1130_nextopen` | (翌始値 - 11:30) / 11:30 × 100 | 昼休み仕込み → 翌朝成り行き売り |
| `target_1230_nextopen` | (翌始値 - 12:30) / 12:30 × 100 | 後場寄り仕込み → 翌朝成り行き売り |

---

## 2. セットアップ

### 必要環境

- Python 3.11 以上
- EDINET API キー（無料取得: https://disclosure2.edinet-fsa.go.jp/）

### インストール手順

```bash
# 1. リポジトリをクローン
git clone https://github.com/yomoriyo1124-code/shikiho-ml-pipeline.git
cd shikiho-ml-pipeline

# 2. 依存ライブラリをインストール
pip install -r requirements.txt

# 3. 環境変数を設定
#    プロジェクトルートに .env ファイルを作成
echo "EDINET_API_KEY=あなたのAPIキー" > .env

# 4. 必要なディレクトリを作成
mkdir -p date/row_date date/processed models reports feedback
```

### requirements.txt の主要パッケージ

| パッケージ | 用途 |
|-----------|------|
| yfinance | 株価・財務データ取得 |
| lightgbm / xgboost / catboost | 勾配ブースティングモデル |
| scikit-learn | Ridge・RandomForest・交差検証 |
| optuna | ハイパーパラメータ自動最適化 |
| jinja2 | HTML レポートテンプレート |
| requests / lxml | EDINET API・XBRL 解析 |

---

## 3. 毎週の使い方

### ステップ 1: 四季報CSVを配置

四季報オンラインからダウンロードした CSV を以下に配置:

```
date/row_date/四季報_260102.csv   ← ファイル名に日付（YYMMDD）が入っていること
```

**CSVに必要な列:** 証券コード, 社名, 市場, 経常修正率(%), 純利益修正率(%), 時価総額(億円), 更新日, 備考, 発表

### ステップ 2: パイプライン実行

```bash
python run_pipeline.py date/row_date/四季報_260102.csv
```

実行完了まで **60〜120分** かかります（Agent2 の機械学習が最も時間がかかります）。

### ステップ 3: 結果を確認

完了後、以下のファイルが生成されます:

```
reports/report_260102.html          ← ブラウザで開く → 上昇期待ランキングを確認
date/processed/四季報_260102_predicted.csv  ← 予測値付きCSV
feedback/feedback_260102.json       ← 次週用（自動で読み込まれる）
```

### 途中から再開する場合

Agent1 が完了している場合、Agent2 以降だけを実行できます:

```bash
# Agent2 以降を実行
python run_pipeline.py --start-from agent2

# Agent3 以降を実行（レポートを再生成したい場合など）
python run_pipeline.py --start-from agent3
```

---

## 4. 各エージェント詳細

---

### Agent 1: データ更新エージェント

**ファイル:** `agents/agent1_data_updater.py`

**何をするか:**
四季報CSV の各銘柄について、発表日の株価5価格と財務指標を取得し、4つのターゲット変数を計算します。

**取得するデータ:**

| データ種別 | ソース | 主な内容 |
|-----------|--------|---------|
| 株価5価格 | yfinance (1h足) | 始値・11:30・12:30・終値・翌日始値 |
| 基本財務指標 | yfinance Ticker.info | PER・ROE・配当利回り・β値 など25項目 |
| 四半期BS | yfinance quarterly_balance_sheet | 現預金・純資産・負債 など |
| キャッシュフロー | EDINET XBRL | 営業CF・投資CF・財務CF・設備投資 |
| 自己資本比率 | EDINET XBRL | 正確な自己資本比率 |

**入力:**
- `date/row_date/四季報_YYMMDD.csv`
- `feedback/feedback_前週YYMMDD.json`（初回は不要）

**出力:**
- `date/processed/四季報_YYMMDD_processed.csv`
- `date/processed/master.csv`（全週分を蓄積）

**単独実行:**
```bash
python agents/agent1_data_updater.py date/row_date/四季報_260102.csv
```

**処理時間の目安:** 1,000銘柄で 30〜60分（yfinance の API レート制限があるため）

---

### Agent 2: ML予測エージェント

**ファイル:** `agents/agent2_ml_predictor.py`

**何をするか:**
蓄積された master.csv を使い、4ターゲットそれぞれについて5つのモデルを比較し、最良モデルで最新週を予測します。

**モデル比較:**

| モデル | 特徴 |
|--------|------|
| Ridge 回帰 | 線形ベースライン |
| Random Forest | 外れ値に強い、ノンパラメトリック |
| LightGBM | 高速、カテゴリ変数対応（主力候補） |
| XGBoost | LightGBM との比較用 |
| CatBoost | 業種・市場などカテゴリ特徴量に強い |

**自動モード切替:**

```
方向一致率 < 55% かつ 訓練週数 >= 4週
    → 5クラス分類モードに切替

5クラスの意味:
  クラス 4 = 強気上昇（+2%超）
  クラス 3 = 軽度上昇（+0.5%〜+2%）
  クラス 2 = 横ばい（±0.5%以内）
  クラス 1 = 軽度下落（-2%〜-0.5%）
  クラス 0 = 強気下落（-2%未満）
```

**入力:** `date/processed/master.csv`

**出力:**
- `date/processed/四季報_YYMMDD_predicted.csv`（予測値・クラス付き）
- `models/model_{target}_{YYMMDD}.pkl` × 4（学習済みモデル）
- `models/model_summary_YYMMDD.json`（精度サマリ）

**単独実行:**
```bash
python agents/agent2_ml_predictor.py
```

**処理時間の目安:** 4ターゲット × Optuna 最適化で 60〜90分

---

### Agent 3: レポート生成エージェント

**ファイル:** `agents/agent3_reporter.py`

**何をするか:**
Agent2 の結果を分析し、7セクション構成の HTML レポートを生成します。ブラウザで開くだけで確認できます。

**レポートの構成:**

| セクション | 内容 |
|-----------|------|
| 1. 精度サマリ | 4ターゲット × MAE・方向一致率・F1 の比較表 |
| 2. 特徴量重要度 | Top 15 グラフ（4ターゲット別） |
| 3. モデル比較表 | 全5モデルの精度一覧 |
| 4. 上昇期待ランキング | Top 30（ランク「軽度上昇」以上の銘柄） |
| 5. 前週予測 vs 実績 | 乖離が大きかった銘柄分析（実績データがある場合） |
| 6. 次週改善サジェスト | モード切替推奨・低重要度特徴量の提示 |

**入力:**
- `date/processed/四季報_YYMMDD_predicted.csv`
- `models/model_summary_YYMMDD.json`

**出力:** `reports/report_YYMMDD.html`

**単独実行:**
```bash
python agents/agent3_reporter.py
# または明示的に指定
python agents/agent3_reporter.py date/processed/四季報_260102_predicted.csv models/model_summary_260102.json
```

---

### Agent 4: フィードバックエージェント

**ファイル:** `agents/agent4_feedback.py`

**何をするか:**
モデルサマリと予測CSVを分析し、翌週の Agent1・Agent2 への改善ヒントを JSON 形式で出力します。次週実行時に自動で読み込まれます。

**自動分析ルール:**

| 分析項目 | ルール |
|---------|--------|
| 推奨モデル | `all_models_reg` の中で方向一致率が最高のモデルを推奨（ridge は分類時除外） |
| 分類/回帰切替 | 方向一致率 < 55% → `switch_to_classification: true` |
| 低重要度特徴量 | 重要度 < 0.5% の特徴量を `low_importance` に列挙 |
| データ品質 | NaN率 30% 超の列を `data_quality_issues` に報告 |
| ベストターゲット | 4ターゲット中で方向一致率が最も高いものを `best_target` に設定 |

**フィードバック JSON の構造:**

```json
{
  "week": "260102",
  "agent1_feedback": {
    "add_features": ["追加推奨の特徴量"],
    "remove_features": ["除外推奨の特徴量"],
    "data_quality_issues": ["NaN率が高い列の警告"]
  },
  "agent2_feedback": {
    "target_1130_close": {
      "recommended_model": "lgbm_classification",
      "switch_to_classification": true,
      "feature_importance_hints": { "top_features": [...], "low_importance": [...] },
      "hyperparameter_hints": { "xgb_n_estimators": 295, ... }
    },
    "...": "...",
    "best_target": "target_1130_close"
  },
  "general_notes": "週次サマリコメント"
}
```

**入力:**
- `models/model_summary_YYMMDD.json`
- `reports/report_YYMMDD.html`（参照のみ）
- `date/processed/四季報_YYMMDD_predicted.csv`（データ品質チェック用）

**出力:** `feedback/feedback_YYMMDD.json`

**単独実行:**
```bash
python agents/agent4_feedback.py
```

---

## 5. 出力ファイルの見方

### HTMLレポート（最重要）

`reports/report_YYMMDD.html` をブラウザで開きます。

**見るべきポイント:**

1. **セクション1「精度サマリ」** — ベストターゲット(★)と方向一致率を確認
   - 方向一致率 55% 以上 → 回帰モードで精度良好
   - 55% 未満 → 分類モード（週数が増えれば改善する）

2. **セクション4「上昇期待ランキング」** — 翌週の投資候補
   - ランク「強気上昇（クラス4）」銘柄を最優先で確認
   - 予測値（数値）が高いほど上昇幅が期待される

3. **セクション5「前週予測 vs 実績」** — モデルの信頼性チェック
   - ランク乖離が小さいほどモデルが安定している

### 予測CSV

`date/processed/四季報_YYMMDD_predicted.csv` には以下の列が追加されています:

| 列名 | 内容 |
|------|------|
| `pred_target_1130_close` | 予測値（回帰値 or クラス確率） |
| `rank_class_target_1130_close` | 予測クラス（0〜4） |
| ※ 4ターゲット分それぞれ存在 | |

---

## 6. ファイル構成

```
四季報/
├── run_pipeline.py              # パイプライン実行スクリプト（毎週これを実行）
├── requirements.txt             # 依存ライブラリ一覧
├── .env                         # EDINET_API_KEY（gitignore済み）
│
├── agents/
│   ├── agent1_data_updater.py   # データ更新エージェント
│   ├── agent2_ml_predictor.py   # ML予測エージェント
│   ├── agent3_reporter.py       # レポート生成エージェント
│   ├── agent4_feedback.py       # フィードバックエージェント
│   └── edinet_client.py         # EDINET API v2 クライアント
│
├── date/
│   ├── row_date/                # 週次CSV配置場所（gitignore済み）
│   │   └── 四季報_YYMMDD.csv
│   └── processed/               # Agent1 の出力
│       ├── 四季報_YYMMDD_processed.csv  # 週次処理済みCSV（gitignore済み）
│       ├── 四季報_YYMMDD_predicted.csv  # Agent2 予測値付きCSV（gitignore済み）
│       └── master.csv                   # 全週蓄積マスタ（gitignore済み）
│
├── models/                      # Agent2 の出力
│   ├── model_summary_YYMMDD.json    # 精度サマリ（バージョン管理済み）
│   └── model_*.pkl                  # 学習済みモデル（gitignore済み）
│
├── reports/                     # Agent3 の出力（gitignore済み）
│   └── report_YYMMDD.html
│
├── feedback/                    # Agent4 の出力
│   └── feedback_YYMMDD.json         # 翌週用フィードバック（バージョン管理済み）
│
└── docs/
    └── plans/                   # 設計計画ドキュメント
        └── 03_add_nextday_targets.md
```

---

## 7. よくあるトラブルと対処法

### Agent1 が途中で止まる

**原因:** yfinance の API レート制限、または銘柄コードが見つからない
**対処:** 取得失敗銘柄はスキップしてログに記録されます。`date/processed/agent1_run_YYMMDD.log` を確認してください。

### EDINET データが取得できない

**原因:** `EDINET_API_KEY` が未設定、または `.env` ファイルが見つからない
**確認:**
```bash
# .env ファイルの確認
cat .env
# → EDINET_API_KEY=xxxxx が表示されればOK
```

### Agent2 の処理が非常に遅い

**原因:** Optuna のハイパーパラメータ探索（50〜100試行）に時間がかかります
**対処:** 訓練週数が少ない初期（8週未満）は試行数が自動で 50回に抑えられます。

### master.csv に重複データが入ってしまった

**対処:**
```bash
# master.csv を削除して Agent1 を再実行（重複なしで再生成されます）
rm date/processed/master.csv
python run_pipeline.py date/row_date/四季報_YYMMDD.csv
```

### レポートの「前週予測 vs 実績」が「データなし」と表示される

**原因:** 正常です。最新週のターゲット実績は翌週以降に確定するため、初回は必ずこの表示になります。

---

## 8. 設計上の注意事項

### データの蓄積とモデル改善の関係

| 蓄積週数 | 状況 |
|---------|------|
| 1〜3週 | Ridge・RandomForest 中心。まだ精度は出にくい |
| 4〜7週 | 精度判定・分類切替が機能し始める |
| 8週以上 | Optuna 試行数が 100回に増加、フィードバックループが有効化 |
| 20週以上 | 季節性・業種傾向が学習できる水準 |

### フィードバック JSON の手動編集

`feedback/feedback_YYMMDD.json` はテキストエディタで直接編集できます。Agent4 の自動判断が不適切だと感じた場合は、手動で修正してから次週の実行を行ってください。

### 1h足データの制約

yfinance の 1h足データは最長 **730日分（約2年）** しか取得できません。2年以上前の発表日については株価データが取得できないため、そのような古い週次CSVには対応していません。

---

## データソース・ライセンス

- **yfinance:** Yahoo Finance データ（非商用利用の範囲で使用）
- **EDINET:** 金融庁 EDINET（公開データ、API利用規約に従う）
- **四季報CSVデータ:** 東洋経済新報社の著作物のためリポジトリには含めていません

---

*最終更新: 2026-02-21*
