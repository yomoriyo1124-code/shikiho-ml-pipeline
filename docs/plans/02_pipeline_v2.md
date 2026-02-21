# 四季報オンライン 株価上昇率予測パイプライン v2（計画修正版）

> **File:** docs/plans/02_pipeline_v2.md
> **Date:** 2026-02-21
> **Status:** Draft
> **前回計画:** docs/plans/01_shikiho_ml_pipeline.md

---

## 1. 背景 / ゴール

**背景:**
01計画からの主な変更点3点：
1. yfinance の **1h足** に切り替え → 歴史データが7日→約730日に拡大（過去2年遡及可能）
2. Agent2 のモデルを **LightGBM を基準に複数モデルを比較探索** するアーキテクチャへ変更
3. **Agent4（フィードバックエージェント）** を追加 → レポート分析結果を Agent1・Agent2 へ翌週分の改善指示として渡す

**ゴール:**
週次CSVを `run_pipeline.py` に渡すだけで、データ拡充→複数モデル比較→レポート生成→翌週フィードバック生成 が自動実行されるパイプラインを構築する。

---

## 2. スコープ

### In（やること）
- [ ] Agent 1: 週次CSVに株価（1h足）・基本情報を付与
- [ ] Agent 2: 複数MLモデルを比較し最良モデルで予測。精度不足なら5クラス分類
- [ ] Agent 3: 予測精度・ランキング・特徴量重要度のHTMLレポート生成
- [ ] Agent 4: レポートを解析し Agent1・Agent2 への翌週フィードバックJSONを生成
- [ ] 4エージェントをスクリプトで直列パイプライン化
- [ ] 1h足ベースで過去データ遡及取得（最大約730日）

### Out（やらないこと）
- リアルタイム自動売買・投資推奨
- 四季報オンラインへの自動スクレイピング（CSVは手動ダウンロード継続）
- 有料API利用
- 深層学習（LSTM等）初期フェーズでは対象外（データ蓄積後に検討）

---

## 3. 現状把握

**CSVカラム（確認済）:**

| カラム名 | 内容 | 備考 |
|----------|------|------|
| 証券コード | 銘柄コード | yfinance: `{コード}.T` |
| 社名 | 会社名 | |
| 市場 | 東P/東G/東S | カテゴリ特徴量 |
| 経常修正率(%) | 経常利益前回比修正率 | 主要特徴量 |
| 純利益修正率(%) | 純利益前回比修正率 | 主要特徴量 |
| 時価総額(億円) | 時価総額 | log変換検討 |
| 更新日 | 四季報更新日 | 発表とのタイムラグ計算 |
| 備考 | 備考欄（多NaN） | |
| 発表 | 株価取得対象日 | |

**yfinance 1h足 検証結果（2026-02-19 7203.T）:**

| バー時刻 | Open | Close | 意味 |
|----------|------|-------|------|
| 11:00 | 3751 | **3766** | **前場引け（11:30）に対応** |
| 12:00 | **3763** | 3755 | **後場始値（12:30）に対応** |
| 15:00 | 3777 | 3764 | 後場引け（終値）に対応 |

- 取得方法:
  - **11:30価格** = 11:00バーの `Close`
  - **12:30価格** = 12:00バーの `Open`
  - **終値** = 日次データの `Close`（精度優先）または 15:00バーの `Close`
- 歴史データ可能期間: **約730日（2023年2月〜）**
  → 週次CSVが複数週ない場合でも yfinance 側の過去株価データは取得できる

**前提・制約:**
- 1h足は最長730日。それ以前の日付は取得不可
- 英字コードを含む場合（例: 135A）も `.T` 付きで yfinance は対応
- 日次終値は yfinance の `history(period='1d', interval='1d')` で取得が高精度

---

## 4. 進め方（作業ステップ）

### Phase A: 環境構築

1. **ライブラリ整備** — `requirements.txt` を作成
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

2. **プロジェクト構造整備**
   ```
   四季報/
   ├── date/
   │   ├── row_date/            # 元CSV（手動配置）
   │   └── processed/           # 拡充済みCSV（蓄積）
   │       └── master.csv       # 全週マージ
   ├── models/
   │   └── model_YYMMDD.pkl     # 学習済みモデル保存
   ├── reports/
   │   └── report_YYMMDD.html   # 週次レポート
   ├── feedback/
   │   └── feedback_YYMMDD.json # Agent4出力（翌週用）
   ├── agents/
   │   ├── agent1_data_updater.py
   │   ├── agent2_ml_predictor.py
   │   ├── agent3_reporter.py
   │   └── agent4_feedback.py
   └── run_pipeline.py
   ```

---

### Phase B: Agent 1 — データ更新エージェント

**入力:**
- `date/row_date/四季報_YYMMDD.csv`（週次CSV）
- `feedback/feedback_前週YYMMDD.json`（Agent4からの翌週フィードバック、初回は空）

**出力:** `date/processed/四季報_YYMMDD_processed.csv` + `date/processed/master.csv` への追記

3. **1h足株価取得モジュール**

   ```python
   # 発表日の1h足を取得
   df_1h = yf.Ticker(f"{code}.T").history(
       start=発表日, end=発表日+1日, interval="1h"
   )
   価格_1130 = df_1h.loc[df_1h.index.hour == 11, 'Close'].iloc[-1]  # 11:00バーClose
   価格_1230 = df_1h.loc[df_1h.index.hour == 12, 'Open'].iloc[0]   # 12:00バーOpen
   終値 = yf.Ticker(f"{code}.T").history(start=発表日, end=発表日+1日)['Close'].iloc[0]
   ```

4. **財務指標取得モジュール** — `Ticker.info` から取得

   | 項目 | yfinanceキー | 備考 |
   |------|-------------|------|
   | 業種 | `sector` / `industry` | 英語→日本語マッピング不要（英語のままカテゴリ化） |
   | 自己資本比率 | `bookValue` / `totalAssets` で計算 または `returnOnEquity`連動 | NaN多い場合は除外 |
   | PER | `trailingPE` または `forwardPE` | 異常値（負・1000超）は除外 |
   | ROE | `returnOnEquity` | |
   | 配当利回り | `dividendYield` | |
   | β値 | `beta` | ボラティリティの代理変数 |

5. **派生特徴量計算**

   | 特徴量 | 計算式 |
   |--------|--------|
   | 修正方向_経常 | `sign(経常修正率)` → -1/0/+1 |
   | 修正幅（絶対値） | `abs(経常修正率)` |
   | 時価総額_log | `log1p(時価総額)` |
   | 更新→発表ラグ(日) | `発表日 - 更新日` |
   | 発表曜日 | `発表日.dayofweek` |
   | 週内累積銘柄数 | 同発表日の累積銘柄数（市場過熱度の代理） |

6. **ターゲット変数計算**
   ```
   target_1130 = (終値 - 価格_1130) / 価格_1130 * 100  [%]
   target_1230 = (終値 - 価格_1230) / 価格_1230 * 100  [%]
   ```

7. **Agent4フィードバック反映** — `feedback_前週.json` の指示があれば特徴量追加・除外を動的に適用

8. **週次ファイル保存 + master.csv への追記**（重複チェック付き）

---

### Phase C: Agent 2 — ML予測エージェント

**入力:**
- `date/processed/master.csv`（蓄積済み全週分）
- `feedback/feedback_前週YYMMDD.json`（Agent4からのモデル変更提案）

**出力:**
- `models/model_YYMMDD.pkl`
- `date/processed/四季報_YYMMDD_predicted.csv`
- `models/model_summary_YYMMDD.json`（精度指標・使用モデル・特徴量重要度）

9. **モデル比較探索**

   回帰モードと分類モード、データ量に応じて自動切替：

   **Step 9-1: ベースライン**
   | モデル | ライブラリ | 特徴 |
   |--------|-----------|------|
   | Ridge回帰 | sklearn | 線形ベースライン、説明性高い |
   | Random Forest | sklearn | ノンパラ・外れ値に強い |

   **Step 9-2: Gradient Boosting 比較**
   | モデル | ライブラリ | 特徴 |
   |--------|-----------|------|
   | **LightGBM** | lightgbm | 高速・カテゴリ変数対応・主力候補 |
   | XGBoost | xgboost | LGBMと比較用 |
   | CatBoost | catboost | カテゴリ特徴量に強い（業種等に有効）|

   **評価指標（回帰）:**
   - MAE（平均絶対誤差）
   - RMSE
   - 方向一致率（上昇/下落を当てた割合）

10. **精度判定 → 分類切替ロジック**

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

    分類評価: F1（macro）、混同行列

11. **時系列交差検証（TimeSeriesSplit n=5）** で全モデルを比較、最良モデルを選択

12. **Optuna ハイパーパラメータ最適化**（最良モデルに対して）
    - 試行回数: 初期50回、データ増加後に100回

13. **最新週への予測適用** → 予測値＋クラス（5ランク）を出力CSVに追記

14. **Agent4フィードバック反映** — モデル変更提案（特徴量追加/除外/モード切替）を適用

---

### Phase D: Agent 3 — レポート分析エージェント

**入力:** 予測済みCSV + `model_summary_YYMMDD.json`
**出力:** `reports/report_YYMMDD.html`

15. **レポート構成**
    - セクション1: 精度サマリー（MAE/RMSE/方向一致率 または F1、前週比）
    - セクション2: 特徴量重要度グラフ（Top 15）
    - セクション3: モデル比較表（全候補の精度一覧）
    - セクション4: 上昇期待ランキング Top 30（銘柄コード・社名・予測値・ランク）
    - セクション5: 前週予測 vs 実績（乖離大きい銘柄の分析）
    - セクション6: 次週改善サジェスト（Agent4への入力メモ）

---

### Phase E: Agent 4 — フィードバックエージェント

**入力:** `reports/report_YYMMDD.html` + `model_summary_YYMMDD.json`
**出力:** `feedback/feedback_YYMMDD.json`

16. **フィードバックJSONの構造**

    ```json
    {
      "week": "YYMMDD",
      "agent1_feedback": {
        "add_features": ["feature_name: 追加理由"],
        "remove_features": ["feature_name: 除外理由"],
        "data_quality_issues": ["issue description"]
      },
      "agent2_feedback": {
        "recommended_model": "lgbm_regression | catboost_regression | lgbm_classification",
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
      "general_notes": "自由記述メモ"
    }
    ```

17. **自動分析ルール（ヒューリスティック）**
    - 特徴量重要度 < 0.5% → `remove_features` に追加
    - 前週 MAE > 閾値 → `switch_to_classification: true`
    - 乖離大銘柄に共通する属性（業種・市場等）があれば `add_features` に提案

18. **フィードバックJSONをAgent1・Agent2が次週実行時に読み込む**（Phase B Step 7 / Phase C Step 14）

---

### Phase F: パイプライン化

19. **`run_pipeline.py` 作成**

    ```bash
    # 使用例（毎週の実行コマンド）
    python run_pipeline.py date/row_date/四季報_260102.csv
    ```

    実行順序:
    ```
    Agent1(data_updater) → Agent2(ml_predictor) → Agent3(reporter) → Agent4(feedback)
    ```

    各エージェントは失敗時にエラーログを出力して停止（次エージェントへは進まない）

---

## 5. 成果物（Deliverables）

| 成果物 | ファイル / 場所 | 形式 | 完了条件 |
|--------|----------------|------|---------|
| 環境定義 | `requirements.txt` | TXT | `pip install -r` が通る |
| データ更新Agent | `agents/agent1_data_updater.py` | Python | 発表日の11:30/12:30/終値が取得され、処理済みCSVが生成される |
| ML予測Agent | `agents/agent2_ml_predictor.py` | Python | 5モデル比較が実行され、最良モデルで予測値付きCSVが生成される |
| レポートAgent | `agents/agent3_reporter.py` | Python | 6セクションのHTMLレポートが出力される |
| フィードバックAgent | `agents/agent4_feedback.py` | Python | `feedback_YYMMDD.json` が生成され、次週のAgent1・2に読み込まれる |
| パイプラインスクリプト | `run_pipeline.py` | Python | 1コマンドで4エージェントが完走する |
| 蓄積マスタCSV | `date/processed/master.csv` | CSV | 週次追記で重複なく蓄積される |
| 週次レポート | `reports/report_YYMMDD.html` | HTML | ブラウザで開ける完成レポート |
| フィードバックJSON | `feedback/feedback_YYMMDD.json` | JSON | 翌週実行時にAgent1・2が正常に読み込める |

---

## 6. リスク / 代替案

| リスク | 影響 | 確率 | 代替案 / 緩和策 |
|--------|------|------|----------------|
| 1h足の11:30/12:30近似がズレる（特に前場引け気配） | Medium | Low | 11:00バーClose / 12:00バーOpen の正確性は検証済み（7203.T）。複数銘柄でも確認を Phase A に含める |
| 1h足も730日を超えた日付は取得不可 | Low | Low | 当面 ~2年分確保できるため許容 |
| モデル比較に時間がかかる（1000銘柄×5モデル） | Medium | Medium | 比較はTrainデータのみ（週次推論時は最良モデルのみ使用）。Optuna試行数を動的調整 |
| Agent4の自動フィードバックが誤った指示を出す | Medium | Medium | JSONはテキストで確認可能。ユーザーが毎週レビューして手動上書きできる設計にする |
| 5クラスのクラス不均衡（横ばいが多数） | Medium | High | `class_weight='balanced'`。評価はmacro-F1を使用 |
| yfinance API レート制限 / タイムアウト | Medium | Medium | `time.sleep(0.5)` を銘柄間に挿入。バッチ分割で再試行ロジック実装 |
| 特定銘柄でティッカーが見つからない | Low | Medium | 取得失敗銘柄はスキップして警告ログに記録。後で手動確認 |

---

## 7. 次アクション

**この計画で次に実行すべき最初の1手:**

> Phase A: `requirements.txt` を作成し環境構築を完了させる。次に、`四季報_260102.csv` の全銘柄コード（上位5件）で 1h足株価取得テストを実行し、11:30・12:30・終値が正しく取得できることを確認する。（英字コード 135A.T の挙動も含めて確認）

**実装順序:**
1. Phase A（環境）→ Phase B（Agent1）→ 動作確認
2. Phase C（Agent2）→ 動作確認
3. Phase D（Agent3）→ 動作確認
4. Phase E（Agent4）→ 動作確認
5. Phase F（パイプライン統合）→ 全体通しテスト

**データ戦略:**
- 初回: `四季報_260102.csv`（2月19日発表分）で Agent1 を実行し、発表日の株価を取得
- 2〜3週目: データ蓄積しながら Agent2 のモデル比較を本格化
- 4週目以降: Agent4のフィードバックループが機能し始める
