"""
Agent1: データ更新エージェント

週次CSVに株価（1h足）・財務指標・派生特徴量・4ターゲット変数を付与して保存する。

入力:
  date/row_date/四季報_YYMMDD.csv
  feedback/feedback_前週YYMMDD.json  （初回は不要）

出力:
  date/processed/四季報_YYMMDD_processed.csv
  date/processed/master.csv

使用例:
  python agents/agent1_data_updater.py date/row_date/四季報_260102.csv
"""

import argparse
import glob
import json
import logging
import math
import os
import re
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# ---- .env 読み込み（agents/ の親ディレクトリを探す）----
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---- agents/ 配下のモジュールを import できるよう sys.path に追加 ----
sys.path.insert(0, str(Path(__file__).resolve().parent))
from edinet_client import EDINETClient  # noqa: E402

warnings.filterwarnings("ignore")

# ---- パス定義 ----
PROCESSED_DIR = Path("date/processed")
FEEDBACK_DIR = Path("feedback")
MASTER_CSV = PROCESSED_DIR / "master.csv"

# ---- EDINET API キー ----
EDINET_API_KEY = os.environ.get("EDINET_API_KEY")

# ---- ロギング ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ============================================================
# ユーティリティ
# ============================================================

def parse_month_day(date_str: str, year: int) -> date | None:
    """日付文字列を date に変換。パース失敗時は None。
    対応フォーマット:
      '2月19日'   → date(2026, 2, 19)
      '2026/2/19' → date(2026, 2, 19)（YYYY/M/D）
    """
    if not date_str or pd.isna(date_str):
        return None
    s = str(date_str).strip()
    # X月Y日 形式
    m = re.match(r"(\d+)月(\d+)日", s)
    if m:
        return date(year, int(m.group(1)), int(m.group(2)))
    # YYYY/M/D 形式
    m = re.match(r"(\d{4})/(\d+)/(\d+)", s)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    logger.warning(f"日付パース失敗: {date_str!r}")
    return None


def extract_year_from_filename(csv_path: str) -> int:
    """ファイル名 四季報_YYMMDD.csv から西暦年を抽出。例: 260102 → 2026"""
    m = re.search(r"_(\d{2})\d{4}", Path(csv_path).stem)
    if m:
        return 2000 + int(m.group(1))
    import datetime
    logger.warning("ファイル名から年を特定できなかった。現在年を使用。")
    return datetime.date.today().year


def load_latest_feedback() -> dict:
    """feedback/ 以下の最新 feedback_*.json を読み込む。なければ空 dict。"""
    files = sorted(glob.glob(str(FEEDBACK_DIR / "feedback_*.json")))
    if not files:
        logger.info("フィードバックファイルなし（初回実行）")
        return {}
    latest = files[-1]
    with open(latest, encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"フィードバック読み込み: {latest}")
    return data


# ============================================================
# Step 4: 株価取得（5価格）
# ============================================================

def fetch_prices(code: str, announce_date: date) -> dict:
    """
    発表日の5価格を yfinance 1h足・日次データから取得する。

    price_open    : 発表日 09:00 バーの Open（始値）
    price_1130    : 発表日 11:00 バーの Close（前場引け近似）
                    ※ 11時バーなし時は前場の最後のバー Close を使用
    price_1230    : 発表日 12:00 バーの Open（後場始値近似）
    price_close   : 発表日終値（日次データ、精度優先）
    price_nextopen: 翌営業日 09:00 バーの Open（週末・祝日自動対応）
    """
    ticker = yf.Ticker(f"{code}.T")

    # ---- 発表日 1h足 ----
    df_1h = ticker.history(
        start=announce_date,
        end=announce_date + timedelta(days=1),
        interval="1h",
    )
    if df_1h.empty:
        raise ValueError("発表日 1h足データなし")

    price_open = float(df_1h.iloc[0]["Open"])

    # price_1130: 11時バー優先、なければ前場の最後のバー（低出来高銘柄対応）
    h11 = df_1h[df_1h.index.hour == 11]
    if not h11.empty:
        price_1130 = float(h11["Close"].iloc[-1])
        price_1130_note = ""
    else:
        pre_noon = df_1h[df_1h.index.hour < 12]
        if pre_noon.empty:
            raise ValueError("前場データなし（price_1130 取得不可）")
        price_1130 = float(pre_noon["Close"].iloc[-1])
        price_1130_note = f"fallback:{pre_noon.index[-1].hour}h"

    # price_1230: 12時バーの Open
    h12 = df_1h[df_1h.index.hour == 12]
    if h12.empty:
        raise ValueError("12時バーなし（price_1230 取得不可）")
    price_1230 = float(h12["Open"].iloc[0])

    # ---- 発表日終値（日次データ） ----
    df_daily = ticker.history(
        start=announce_date,
        end=announce_date + timedelta(days=1),
    )
    if df_daily.empty:
        raise ValueError("発表日 日次データなし")
    price_close = float(df_daily["Close"].iloc[0])

    # ---- 翌営業日始値（最大7日先まで探索） ----
    df_next = ticker.history(
        start=announce_date + timedelta(days=1),
        end=announce_date + timedelta(days=7),
        interval="1h",
    )
    if df_next.empty:
        raise ValueError("翌営業日 1h足データなし（price_nextopen 取得不可）")
    price_nextopen = float(df_next.iloc[0]["Open"])
    next_trade_date = df_next.index[0].date()

    return {
        "price_open":      round(price_open, 2),
        "price_1130":      round(price_1130, 2),
        "price_1230":      round(price_1230, 2),
        "price_close":     round(price_close, 2),
        "price_nextopen":  round(price_nextopen, 2),
        "next_trade_date": str(next_trade_date),
        "price_1130_note": price_1130_note,
    }


# ============================================================
# Step 5: 財務指標取得（Ticker.info）
# ============================================================

def fetch_financials(code: str) -> dict:
    """
    Ticker.info から財務指標を取得する。
    取得失敗・欠損時は None（カラムは保持される）。

    取得項目:
      株式規模   : market_cap, shares_outstanding, float_shares, avg_volume, avg_volume_10d
      バリュエーション: per (trailingPE→forwardPE fallback), pbr, beta
      配当       : dividend_rate, dividend_yield, payout_ratio
      財務健全性 : roe, debt_to_equity (※ 自己資本比率の代替), equity_ratio_approx
      利益率     : profit_margin, operating_margin, gross_margin
      成長率     : revenue_growth, earnings_growth
      テクニカル : week52_high, week52_low, ma50, ma200
      業種       : sector, industry
    """
    _EMPTY_KEYS = [
        "sector", "industry",
        "market_cap", "shares_outstanding", "float_shares",
        "avg_volume", "avg_volume_10d",
        "per", "pbr", "beta",
        "dividend_rate", "dividend_yield", "payout_ratio",
        "roe", "debt_to_equity", "equity_ratio_approx",
        "profit_margin", "operating_margin", "gross_margin",
        "revenue_growth", "earnings_growth",
        "week52_high", "week52_low", "ma50", "ma200",
    ]
    empty = {k: None for k in _EMPTY_KEYS}

    try:
        info = yf.Ticker(f"{code}.T").info
    except Exception as e:
        logger.warning(f"[{code}] Ticker.info 取得失敗: {e}")
        return empty

    def safe(key: str):
        v = info.get(key)
        return v if v not in (None, float("inf"), float("-inf")) else None

    def flt(key: str, digits: int = 4):
        v = safe(key)
        return round(float(v), digits) if v is not None else None

    def pct(key: str, digits: int = 4):
        """yfinance の 0-1 小数値を % 表記に変換（例: 0.05 → 5.0）"""
        v = safe(key)
        return round(float(v) * 100, digits) if v is not None else None

    # PER: 異常値（負・1000超）を除外
    per_raw = safe("trailingPE") or safe("forwardPE")
    per = round(float(per_raw), 2) if per_raw and 0 < per_raw < 1000 else None

    # 自己資本比率の近似値: D/E比から逆算（※ 有利子負債ベースの近似）
    # debt_to_equity は yfinance では % 表記（例: 45.6 = 45.6%）
    de = safe("debtToEquity")
    equity_ratio_approx = None
    if de is not None and de >= 0:
        equity_ratio_approx = round(100.0 / (1.0 + de / 100.0), 4)

    return {
        # 業種
        "sector":               safe("sector"),
        "industry":             safe("industry"),
        # 株式規模
        "market_cap":           safe("marketCap"),
        "shares_outstanding":   safe("sharesOutstanding"),
        "float_shares":         safe("floatShares"),
        "avg_volume":           safe("averageVolume"),
        "avg_volume_10d":       safe("averageVolume10days"),
        # バリュエーション
        "per":                  per,
        "pbr":                  flt("priceToBook", 4),
        "beta":                 flt("beta", 4),
        # 配当
        "dividend_rate":        flt("dividendRate", 4),
        "dividend_yield":       flt("dividendYield", 4),   # yfinance は既に % 表記（例: 2.84 = 2.84%）
        "payout_ratio":         pct("payoutRatio", 4),
        # 財務健全性
        "roe":                  pct("returnOnEquity", 4),
        "debt_to_equity":       flt("debtToEquity", 4),   # % 表記（yfinance仕様）
        "equity_ratio_approx":  equity_ratio_approx,      # ≈1/(1+D/E) の近似値
        # 利益率（0-1 → %）
        "profit_margin":        pct("profitMargins", 4),
        "operating_margin":     pct("operatingMargins", 4),
        "gross_margin":         pct("grossMargins", 4),
        # 成長率（0-1 → %）
        "revenue_growth":       pct("revenueGrowth", 4),
        "earnings_growth":      pct("earningsGrowth", 4),
        # テクニカル
        "week52_high":          flt("fiftyTwoWeekHigh", 2),
        "week52_low":           flt("fiftyTwoWeekLow", 2),
        "ma50":                 flt("fiftyDayAverage", 2),
        "ma200":                flt("twoHundredDayAverage", 2),
    }


# ============================================================
# Step 5b: yfinance quarterly Balance Sheet（Option A）
# ============================================================

def fetch_quarterly_bs(code: str) -> dict:
    """
    quarterly_balance_sheet から追加財務指標を取得する（CF は yfinance では空のため BS のみ）。

    取得項目:
      exact_equity_ratio_a : 自己資本比率（正確値、equity/assets）
      current_ratio        : 流動比率（流動資産/流動負債）
      total_assets_m       : 総資産（百万円）
      net_assets_m         : 純資産（百万円）
      total_debt_m         : 有利子負債合計（百万円）
      net_debt_m           : ネット有利子負債（百万円）
      working_capital_m    : 運転資本（百万円）
      cash_m               : 現金・現金同等物（百万円）
    """
    _KEYS = [
        "exact_equity_ratio_a", "current_ratio",
        "total_assets_m", "net_assets_m", "total_debt_m",
        "net_debt_m", "working_capital_m", "cash_m",
    ]
    empty = {k: None for k in _KEYS}

    try:
        bs = yf.Ticker(f"{code}.T").quarterly_balance_sheet
        if bs is None or bs.empty:
            return empty

        latest = bs.iloc[:, 0]

        def get_val(*keys):
            for k in keys:
                if k in latest.index and pd.notna(latest[k]):
                    return float(latest[k])
            return None

        def to_m(v):
            return round(v / 1_000_000, 2) if v is not None else None

        total_assets = get_val("Total Assets")
        equity = get_val("Common Stock Equity", "Stockholders Equity", "Total Equity Gross Minority Interest")
        current_assets = get_val("Current Assets")
        current_liabilities = get_val("Current Liabilities")

        exact_equity_ratio = None
        if total_assets and equity and total_assets != 0:
            exact_equity_ratio = round(equity / total_assets * 100, 4)

        current_ratio = None
        if current_assets and current_liabilities and current_liabilities != 0:
            current_ratio = round(current_assets / current_liabilities, 4)

        return {
            "exact_equity_ratio_a": exact_equity_ratio,
            "current_ratio":        current_ratio,
            "total_assets_m":       to_m(total_assets),
            "net_assets_m":         to_m(equity),
            "total_debt_m":         to_m(get_val("Total Debt")),
            "net_debt_m":           to_m(get_val("Net Debt")),
            "working_capital_m":    to_m(get_val("Working Capital")),
            "cash_m":               to_m(get_val(
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments",
            )),
        }

    except Exception as e:
        logger.warning(f"[{code}] quarterly_balance_sheet 取得失敗: {e}")
        return empty


# ============================================================
# Step 6: 派生特徴量計算
# ============================================================

def calc_derived_features(
    row: dict,
    announce_date: date,
    update_date: date | None,
    cumulative_count: int,
) -> dict:
    """CSVの基本情報から派生特徴量を計算する。"""

    def to_float(val) -> float | None:
        try:
            return float(str(val).replace(",", "").strip())
        except (ValueError, TypeError):
            return None

    keijo = to_float(row.get("経常修正率(%)"))
    jika = to_float(row.get("時価総額(億円)"))

    lag_days = (announce_date - update_date).days if update_date else None

    return {
        "修正方向_経常":   int(math.copysign(1, keijo)) if keijo and keijo != 0 else 0,
        "修正幅_絶対値":   round(abs(keijo), 4) if keijo is not None else None,
        "時価総額_log":    round(math.log1p(jika), 6) if jika and jika > 0 else None,
        "更新発表ラグ_日": lag_days,
        "発表曜日":        announce_date.weekday(),  # 0=月, 6=日
        "週内累積銘柄数":  cumulative_count,
    }


# ============================================================
# Step 7: 4ターゲット変数計算
# ============================================================

def calc_targets(prices: dict) -> dict:
    """5価格から4ターゲット変数（上昇率%）を計算する。"""
    p1130 = prices["price_1130"]
    p1230 = prices["price_1230"]
    pclose = prices["price_close"]
    pnext = prices["price_nextopen"]
    return {
        "target_1130_close":    round((pclose - p1130) / p1130 * 100, 4),
        "target_1230_close":    round((pclose - p1230) / p1230 * 100, 4),
        "target_1130_nextopen": round((pnext - p1130) / p1130 * 100, 4),
        "target_1230_nextopen": round((pnext - p1230) / p1230 * 100, 4),
    }


# ============================================================
# Step 8: Agent4 フィードバック反映
# ============================================================

def apply_feedback(df: pd.DataFrame, feedback: dict) -> pd.DataFrame:
    """agent1_feedback の remove_features を DataFrame に適用する。"""
    a1 = feedback.get("agent1_feedback", {})

    for item in a1.get("remove_features", []):
        col = item.split(":")[0].strip()
        if col in df.columns:
            df = df.drop(columns=[col])
            logger.info(f"フィードバック: カラム削除 '{col}'")

    for issue in a1.get("data_quality_issues", []):
        logger.warning(f"データ品質注意: {issue}")

    for item in a1.get("add_features", []):
        logger.info(f"フィードバック: 追加提案 '{item}'（手動実装が必要）")

    return df


# ============================================================
# Step 9: master.csv 更新
# ============================================================

def update_master(processed_df: pd.DataFrame) -> None:
    """processed_df を master.csv に追記する（証券コード×発表 で重複除外）。"""
    if MASTER_CSV.exists():
        master_df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig")
        before = len(master_df)
        combined = pd.concat([master_df, processed_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["証券コード", "発表"], keep="first")
        added = len(combined) - before
        logger.info(f"master.csv: {before} 行 → {len(combined)} 行（+{added} 行追加）")
    else:
        combined = processed_df.copy()
        logger.info(f"master.csv: 新規作成 {len(combined)} 行")

    combined.to_csv(MASTER_CSV, index=False, encoding="utf-8-sig")


# ============================================================
# メイン処理
# ============================================================

def main(csv_path: str) -> None:
    logger.info(f"=== Agent1 開始: {csv_path} ===")

    csv_obj = Path(csv_path)
    if not csv_obj.exists():
        logger.error(f"ファイルが見つかりません: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    year = extract_year_from_filename(csv_path)
    logger.info(f"銘柄数: {len(df)}  /  推定年: {year}")

    feedback = load_latest_feedback()

    # ---- 日付パース & 週内累積カウント ----
    df["_announce_date"] = df["発表"].apply(lambda x: parse_month_day(x, year))
    df["_update_date"] = df["更新日"].apply(
        lambda x: parse_month_day(x, year) if pd.notna(x) else None
    )
    df["_cumcount"] = df.groupby("_announce_date").cumcount() + 1

    # ---- EDINET クライアント & 書類インデックス（一括スキャン）----
    edinet_client = None
    edinet_index: dict = {}
    if EDINET_API_KEY:
        try:
            edinet_client = EDINETClient(EDINET_API_KEY)
            # 全銘柄共通の発表日を取得（CSV の _announce_date の最大値を基準に使用）
            base_date = df["_announce_date"].dropna().max()
            if base_date:
                logger.info("EDINET 書類インデックスを構築中...")
                edinet_index = edinet_client.build_index_for_period(base_date)
        except Exception as e:
            logger.warning(f"EDINET 初期化失敗（スキップ）: {e}")
            edinet_client = None
    else:
        logger.warning("EDINET_API_KEY が未設定のため EDINET 取得をスキップ")

    # ---- 銘柄ループ ----
    results = []
    skip_count = 0

    for idx, row in df.iterrows():
        code = str(row["証券コード"]).strip()
        name = str(row["社名"]).strip()
        announce_date = row["_announce_date"]
        update_date = row["_update_date"]
        cumcount = int(row["_cumcount"])

        if announce_date is None:
            logger.warning(f"[{code}] 発表日パース失敗、スキップ")
            skip_count += 1
            continue

        logger.info(f"[{idx+1}/{len(df)}] {code} {name}  発表日={announce_date}")

        try:
            prices = fetch_prices(code, announce_date)
            if prices["price_1130_note"]:
                logger.warning(f"  [{code}] price_1130 フォールバック: {prices['price_1130_note']}")

            # ---- Option A: yfinance info + quarterly BS ----
            financials = fetch_financials(code)
            quarterly_bs = fetch_quarterly_bs(code)

            # ---- Option B: EDINET XBRL ----
            edinet_data: dict = {}
            if edinet_client:
                try:
                    edinet_data = edinet_client.get_company_financials(
                        code, announce_date, doc_index=edinet_index
                    )
                except Exception as e:
                    logger.warning(f"  [{code}] EDINET 取得失敗（スキップ）: {e}")

            # ---- A→B 優先でマージ ----
            # 自己資本比率: EDINET > quarterly BS 計算値 > info D/E 近似
            equity_ratio_exact = (
                edinet_data.get("equity_ratio_exact")
                or quarterly_bs.get("exact_equity_ratio_a")
            )

            extra = {
                # quarterly BS 由来
                **quarterly_bs,
                # EDINET 由来（存在すれば上書き）
                **{k: edinet_data[k] for k in edinet_data if k != "equity_ratio_exact"},
                # 最終的な自己資本比率（正確値）
                "equity_ratio_exact": equity_ratio_exact,
            }

            derived = calc_derived_features(row.to_dict(), announce_date, update_date, cumcount)
            targets = calc_targets(prices)

            record = {**row.to_dict(), **prices, **financials, **extra, **derived, **targets}
            for k in ["_announce_date", "_update_date", "_cumcount"]:
                record.pop(k, None)

            results.append(record)

        except Exception as e:
            logger.warning(f"  [{code}] スキップ: {e}")
            skip_count += 1

        time.sleep(0.5)

    # ---- 後処理 ----
    if not results:
        logger.error("処理成功銘柄が0件のため中断。")
        sys.exit(1)

    processed_df = pd.DataFrame(results)
    for c in ["_announce_date", "_update_date", "_cumcount"]:
        if c in processed_df.columns:
            processed_df = processed_df.drop(columns=[c])

    processed_df = apply_feedback(processed_df, feedback)

    # ---- 保存 ----
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{csv_obj.stem}_processed.csv"
    processed_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"処理済みCSV保存: {out_path}")

    update_master(processed_df)

    ok = len(results)
    total = len(df)
    logger.info(
        f"=== Agent1 完了: {ok}/{total} 件成功, {skip_count} 件スキップ ==="
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent1: データ更新エージェント")
    parser.add_argument("csv_path", help="週次CSVファイルのパス（例: date/row_date/四季報_260102.csv）")
    args = parser.parse_args()
    main(args.csv_path)
